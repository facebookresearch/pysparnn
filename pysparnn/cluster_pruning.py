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

def k_best(tuple_list, k, return_metric):
    """Get the k-best tuples by distance.
    Args:
        tuple_list: List of tuples. (distance, value)
        k: Number of tuples to return.
        return_metric: Boolean value indicating if metric values should be
            returned.
    Returns:
        The K-best tuples (distance, value) by distance score.
    """
    tuple_lst = sorted(tuple_list, key=lambda x: x[0],
                       reverse=False)[:k]
    if return_metric:
        return tuple_lst
    else:
        return [x[1] for x in tuple_lst]

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

            This breaks up one O(K) search into two O(sqrt(K)) searches which
            is much much faster when K is big.
    """
    def __init__(self, sparse_features, records_data,
                 distance_type=pysparnn.matrix_distance.CosineDistance):
        """Create a search index composed of recursively defined sparse
        matricies.

        Args:
            sparse_features: A csr_matrix with rows that represent records
                (corresponding to the elements in records_data) and columns
                that describe a point in space for each row.
            records_data: Data to return when a doc is matched. Index of
                corresponds to records_features.
            distance_type: Class that defines the distance measure to use.
        """

        # self.sparse_features = sparse_features
        records_data = np.array(records_data)

        # could make this recursive at the cost of recall accuracy
        # keeping to a single layer for simplicity/accuracy
        num_clusters = int(math.sqrt(sparse_features.shape[0]))
        records_index = np.arange(sparse_features.shape[0])
        clusters_selection = random.sample(records_index, num_clusters)
        clusters_selection = sparse_features[clusters_selection]

        item_to_clusters = collections.defaultdict(list)

        root = distance_type(clusters_selection,
                               np.arange(clusters_selection.shape[0]))

        rng_step = 10000
        for rng in range(0, sparse_features.shape[0], rng_step):
            max_rng = min(rng + rng_step, sparse_features.shape[0])
            records_rng = sparse_features[rng:max_rng]
            for i, clstrs in enumerate(root.nearest_search(records_rng, k=1)):
                for _, cluster in clstrs:
                    item_to_clusters[cluster].append(i + rng)

        self.clusters = []
        cluster_keeps = []
        for k, clust_sel in enumerate(clusters_selection):
            clustr = item_to_clusters[k]
            if len(clustr) > 0:
                mtx = distance_type(vstack(sparse_features[clustr]),
                                      records_data[clustr])
                self.clusters.append(mtx)
                cluster_keeps.append(clust_sel)

        cluster_keeps = vstack(cluster_keeps)
        self.root = distance_type(cluster_keeps,
                                    np.arange(cluster_keeps.shape[0]))

    def search(self, sparse_features, k=1, min_distance=None,
               max_distance=None, k_clusters=1, return_metric=True):
        """Find the closest item(s) for each feature_list in.

        Args:
            sparse_features: A csr_matrix with rows that represent records
                (corresponding to the elements in records_data) and columns
                that describe a point in space for each row.
            k: Return the k closest results.
            min_distance: Return items at least min_distance away from the
                query point.
            max_distance: Return items no more than max_distance away from the
                query point.
            k_clusters: number of clusters to search. This increases recall at
                the cost of some speed.
            return_metric: Return metric values?

        Returns:
            For each element in features_list, return the k-nearest items
            and (optionally) their distance
            [[(score1_1, item1_1), ..., (score1_k, item1_k)],
             [(score2_1, item2_1), ..., (score2_k, item2_k)], ...]

             Note: if return_metric is False then only items are returned
                and not as a tuple.
        """
        # could make this recursive at the cost of recall accuracy
        # should batch requests to clusters to make this more efficent
        ret = []

        nearest = self.root.nearest_search(sparse_features, k=k_clusters,
                                           min_distance=min_distance)

        for i, nearest_clusters in enumerate(nearest):
            curr_ret = []

            for _, cluster in nearest_clusters:

                cluster_items = self.clusters[cluster].\
                        nearest_search(sparse_features[i], k=k,
                                       min_distance=min_distance,
                                       max_distance=max_distance)

                for elements in cluster_items:
                    if len(elements) > 0:
                        if return_metric:
                            curr_ret.extend(elements)
                        else:
                            curr_ret.extend(elements)
            ret.append(k_best(curr_ret, k, return_metric))
        return ret
