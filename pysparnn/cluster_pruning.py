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
import math
import random
import numpy as np
from scipy.sparse import vstack
import pysparnn.matrix_distance

def k_best(tuple_list, k):
    """Get the k-best tuples by distance.
    Args:
        tuple_list: List of tuples. (distance, value)
        k: Number of tuples to return.
    Returns:
        The K-best tuples (distance, value) by distance score.
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
    def __init__(self, sparse_features, records_data,
                 distance_type=pysparnn.matrix_distance.CosineDistance,
                 matrix_size=None):
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
        num_records = sparse_features.shape[0]

        if matrix_size is None:
            matrix_size = int(np.sqrt(num_records))

        num_levels = int(np.ceil(np.log(num_records)/np.log(matrix_size)))

        if num_levels <= 1:
            self.is_terminal = True
            self.root = distance_type(sparse_features,
                                 records_data)
        else:
            self.is_terminal = False 
            records_data = np.array(records_data)

            records_index = np.arange(sparse_features.shape[0])
            clusters_size = min(matrix_size, num_records)
            clusters_selection = random.sample(records_index, clusters_size)
            clusters_selection = sparse_features[clusters_selection]

            item_to_clusters = collections.defaultdict(list)

            root = distance_type(clusters_selection,
                                   np.arange(clusters_selection.shape[0]))

            rng_step = matrix_size
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
                    index = ClusterIndex(
                                vstack(sparse_features[clustr]), 
                                records_data[clustr],
                                distance_type=distance_type,
                                matrix_size=matrix_size)
                    clusters.append(index)
                    cluster_keeps.append(clust_sel)

            cluster_keeps = vstack(cluster_keeps)
            clusters = np.array(clusters)
            
            self.root = distance_type(cluster_keeps, clusters)

    def search(self, sparse_features, k=1, min_distance=None,
               max_distance=None, k_clusters=1):
        """Find the closest item(s) for each feature_list in.

        Args:
            sparse_features: A csr_matrix with rows that represent records
                (corresponding to the elements in records_data) and columns
                that describe a point in space for each row.
            k: Return the k closest results.
            min_distance: Return items at least min_distance away from the
                query point. Defaults to any distance. 
            max_distance: Return items no more than max_distance away from the
                query point. Defaults to any distance.
            k_clusters: number of branches (clusters) to search at each level. 
                This increases recall at the cost of some speed. 
                
                Note: min_distance, max_distance constraints are also applied.
                    This means there may be less than k_clusters searched at 
                    each level. We are garunteed to always search at least the 
                    closest branch above min_distance. Further elements are 
                    added so long as the k_clusters and max_distance checks 
                    pass.

                    This means each search will fully traverse at least one 
                    (but at most k_clusters) clusters at each level. 

        Returns:
            For each element in features_list, return the k-nearest items
            and their distance score
            [[(score1_1, item1_1), ..., (score1_k, item1_k)],
             [(score2_1, item2_1), ..., (score2_k, item2_k)], ...]
        """
        if self.is_terminal:
            ret = self.root.nearest_search(sparse_features, k=k,
                                                  min_distance=min_distance,
                                                  max_distance=max_distance)
        else:
            ret = []
            nearest = self.root.nearest_search(sparse_features, k=k_clusters)
            
            for i, nearest_clusters in enumerate(nearest):
                curr_ret = []
                for distance, cluster in nearest_clusters:

                    # skip over entries that are not within the distance
                    # threshold but always return the closest branch
                    empty_results = not len(curr_ret) == 0
                    min_distance_fail = distance < min_distance
                    max_distance_fail = distance > max_distance
                    if (empty_results or min_distance_fail) and\
                            max_distance_fail:
                        continue

                    cluster_items = cluster.\
                            search(sparse_features[i], k=k,
                                   k_clusters=k_clusters,
                                   min_distance=min_distance,
                                   max_distance=max_distance)

                    for elements in cluster_items:
                        if len(elements) > 0:
                            curr_ret.extend(elements)
                ret.append(k_best(curr_ret, k))
        return ret
