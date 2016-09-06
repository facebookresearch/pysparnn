# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Defines a cluster pruing search structure to do sparse K-NN Queries"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
import random
import numpy as np
from scipy.sparse import vstack
import pysparnn.matrix_distance

def k_best(tuple_list, k):
    """Get the k-best tuples by distance.
    Args:
        tuple_list: List of tuples. (distance, value)
        k: Number of tuples to return.
    """
    tuple_lst = sorted(tuple_list, key=lambda x: x[0],
                       reverse=False)[:k]

    return tuple_lst

class ClusterIndex(object):
    """ Search structure which gives speedup at slight loss of recall.

        Uses cluster pruning structure as defined in:
        http://nlp.stanford.edu/IR-book/html/htmledition/cluster-pruning-1.html

        tldr - searching for a document in an index of K documents is naievely
            O(K). However you can create a tree structure where the first level
            is O(sqrt(K)) and each of the leaves are also O(sqrt(K)).

            You randomly pick sqrt(K) items to be in the top level. Then for
            the K doccuments you assign it to the closest neighbor in the top
            level.

            This breaks up one O(K) search into O(2 * sqrt(K)) searches which
            is much much faster when K is big.

            This generalizes to h levels. The runtime becomes:
                O(h * h_root(K))
    """
    def insert(self, sparse_feature, record):
        """Insert a single record into the index.
        
        Args:
            sparse_feature: sparse feature vector
            record: record to return as the result of a search
        """
        
        nearest = self
        while not nearest.is_terminal:
            nearest = nearest.root.nearest_search(sparse_feature, k=1)
            _, nearest = nearest[0][0]

        cluster_index = nearest
        parent_index = cluster_index.parent
        while parent_index and cluster_index.matrix_size * 2 < \
                len(cluster_index.root.get_records()):
            cluster_index = parent_index
            parent_index = cluster_index.parent
       
        cluster_index.reindex(sparse_feature, record)

        

    def _get_child_data(self):
        if self.is_terminal:
            return [self.root.get_feature_matrix()], [self.root.get_records()]
        else:
            result_features = []
            result_records = []
    
            for c in self.root.get_records():
                features, records = c._get_child_data()

                result_features.extend(features)
                result_records.extend(records)
    
            return result_features, result_records 
    
    def reindex(self, sparse_feature=None, record=None):
        features, records = self._get_child_data()

        flat_rec = []
        for x in records:
            flat_rec.extend(x)

        if sparse_feature <> None and record <> None:
            features.append(sparse_feature)
            flat_rec.append(record)

        self.__init__(vstack(features), flat_rec, self.distance_type, 
                self.desired_matrix_size, self.parent)

    def __init__(self, sparse_features, records_data,
                 distance_type=pysparnn.matrix_distance.CosineDistance,
                 matrix_size=None,
                 parent=None):
        """Create a search index composed of recursively defined sparse
        matricies.

        Args:
            sparse_features: A csr_matrix with rows that represent records
                (corresponding to the elements in records_data) and columns
                that describe a point in space for each row.
            records_data: Data to return when a doc is matched. Index of
                corresponds to records_features.
            distance_type: Class that defines the distance measure to use.
            matrix_size: Ideal size for matrix multiplication. This controls
                the depth of the tree. Defaults to 2 levels (approx).
        """

        self.is_terminal = False
        self.parent = parent
        self.distance_type = distance_type
        self.desired_matrix_size = matrix_size
        num_records = sparse_features.shape[0]

        if matrix_size is None:
            matrix_size = max(int(np.sqrt(num_records)), 100)
        else:
            matrix_size = int(matrix_size)

        self.matrix_size = matrix_size

        num_levels = np.log(num_records)/np.log(self.matrix_size)

        if num_levels <= 1.4:
            self.is_terminal = True
            self.root = distance_type(sparse_features,
                                      records_data)
        else:
            self.is_terminal = False
            records_data = np.array(records_data)

            records_index = np.arange(sparse_features.shape[0])
            clusters_size = min(self.matrix_size, num_records)
            clusters_selection = random.sample(records_index, clusters_size)
            clusters_selection = sparse_features[clusters_selection]

            item_to_clusters = collections.defaultdict(list)

            root = distance_type(clusters_selection,
                                 np.arange(clusters_selection.shape[0]))

            rng_step = self.matrix_size
            for rng in range(0, sparse_features.shape[0], rng_step):
                max_rng = min(rng + rng_step, sparse_features.shape[0])
                records_rng = sparse_features[rng:max_rng]
                for i, clstrs in enumerate(root.nearest_search(records_rng, k=1)):
                    for _, cluster in clstrs:
                        item_to_clusters[cluster].append(i + rng)

            clusters = []
            cluster_keeps = []
            for k, clust_sel in enumerate(clusters_selection):
                clustr = item_to_clusters[k]
                if len(clustr) > 0:
                    index = ClusterIndex(vstack(sparse_features[clustr]),
                                         records_data[clustr],
                                         distance_type=distance_type,
                                         matrix_size=self.matrix_size, 
                                         parent=self)
                    clusters.append(index)
                    cluster_keeps.append(clust_sel)

            cluster_keeps = vstack(cluster_keeps)
            clusters = np.array(clusters)

            self.root = distance_type(cluster_keeps, clusters)

    def _search(self, sparse_features, k=1, 
                max_distance=None, k_clusters=1):
        """Find the closest item(s) for each feature_list in.

        Args:
            sparse_features: A csr_matrix with rows that represent records
                (corresponding to the elements in records_data) and columns
                that describe a point in space for each row.
            k: Return the k closest results.
            max_distance: Return items no more than max_distance away from the
                query point. Defaults to any distance.
            k_clusters: number of branches (clusters) to search at each level.
                This increases recall at the cost of some speed.

                Note: max_distance constraints are also applied.
                    This means there may be less than k_clusters searched at
                    each level. 
                    This means each search will fully traverse at least one
                    (but at most k_clusters) clusters at each level.

        Returns:
            For each element in features_list, return the k-nearest items
            and their distance score
            [[(score1_1, item1_1), ..., (score1_k, item1_k)],
             [(score2_1, item2_1), ..., (score2_k, item2_k)], ...]
        """
        if self.is_terminal:
            return self.root.nearest_search(sparse_features, k=k,
                                            max_distance=max_distance)
        else:
            ret = []
            nearest = self.root.nearest_search(sparse_features, k=k_clusters)

            for i, nearest_clusters in enumerate(nearest):
                curr_ret = []
                for distance, cluster in nearest_clusters:

                    cluster_items = cluster.\
                            search(sparse_features[i], k=k,
                                   k_clusters=k_clusters,
                                   max_distance=max_distance)

                    for elements in cluster_items:
                        if len(elements) > 0:
                            curr_ret.extend(elements)
                ret.append(k_best(curr_ret, k))
            return ret

    def search(self, sparse_features, k=1, max_distance=None, k_clusters=1, 
            return_distance=True):
        """Find the closest item(s) for each feature_list in.

        Args:
            sparse_features: A csr_matrix with rows that represent records
                (corresponding to the elements in records_data) and columns
                that describe a point in space for each row.
            k: Return the k closest results.
            max_distance: Return items no more than max_distance away from the
                query point. Defaults to any distance.
            k_clusters: number of branches (clusters) to search at each level.
                This increases recall at the cost of some speed.

                Note: max_distance constraints are also applied.
                    This means there may be less than k_clusters searched at
                    each level. 

                    This means each search will fully traverse at least one
                    (but at most k_clusters) clusters at each level.

        Returns:
            For each element in features_list, return the k-nearest items
            and (optionally) their distance score
            [[(score1_1, item1_1), ..., (score1_k, item1_k)],
             [(score2_1, item2_1), ..., (score2_k, item2_k)], ...]

            Note: if return_distance == False then the scores are omitted
            [[item1_1, ..., item1_k],
             [item2_1, ..., item2_k], ...]


        """
        
        # search no more than 1k records at once
        # helps keap the matrix multiplies small
        batch_size = 1000
        results = []
        rng_step = batch_size
        for rng in range(0, sparse_features.shape[0], rng_step):
            max_rng = min(rng + rng_step, sparse_features.shape[0])
            records_rng = sparse_features[rng:max_rng]

            results.extend(self._search(sparse_features=records_rng,
                                        k=k,
                                        max_distance=max_distance,
                                        k_clusters=k_clusters))

        if return_distance:
            return results
        else:
            no_distance = []
            for result in results:
                no_distance.append([x for y, x in result])
            return no_distance

    def _print_structure(self, tabs=''):
        print(tabs + str(self.root.matrix.shape[0]))
        if not self.is_terminal:
            for index in self.root.records_data:
                index.print_structure(tabs + '  ')

    def _max_depth(self):
        if not self.is_terminal:
            max_dep = 0
            for index in self.root.records_data:
                max_dep = max(max_dep, index._max_depth())
            return 1 + max_dep
        else:
            return 1

    def _matrix_sizes(self, ret=None):
        if ret is None:
            ret = []
        ret.append(len(self.root.records_data))
        if not self.is_terminal:
            for index in self.root.records_data:
                ret.extend(index._matrix_sizes())
        return ret
