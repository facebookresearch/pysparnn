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
import pysparnn.matrix_similarity

def k_best(tuple_list, k, return_metric, is_similarity):
    """Get the k-best tuples by similarity.
    Args:
        tuple_list: List of tuples. (similarity, value)
        k: Number of tuples to return.
        return_metric: Boolean value indicating if metric values should be
            returned.
        is_similarity: Boolean value indicating if the metric is a similarity 
            measure (1 meaning similar and 0 meaning different) or a distance.
    Returns:
        The K-best tuples (similarity, value) by similarity score.
    """
    tuple_lst = sorted(tuple_list, key=lambda x: x[0], 
                       reverse=is_similarity)[:k]
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
    def __init__(self, records_features, records_data,
                 similarity_type=pysparnn.matrix_similarity.CosineSimilarity):
        """Create a search index composed of recursively defined sparse
        matricies.

        Args:
            records_features: List of features in the format of
               {feature_name1 -> value1, feature_name2->value2, ...}.
            records_data: Data to return when a doc is matched. Index of
                corresponds to records_features.
            similarity_class: Class that defines the similarity measure to use.
        """

        self.records_features = np.array(records_features)
        self.records_data = np.array(records_data)

        # could make this recursive at the cost of recall accuracy
        # keeping to a single layer for simplicity/accuracy
        num_clusters = int(math.sqrt(len(self.records_features)))
        clusters_selection = random.sample(self.records_features, num_clusters)

        item_to_clusters = collections.defaultdict(list)

        root = similarity_type(clusters_selection,
                               list(range(len(clusters_selection))))

        rng_step = 10000
        for rng in range(0, len(records_features), rng_step):
            records_rng = records_features[rng:rng + rng_step]
            for i, clstrs in enumerate(root.nearest_search(records_rng, k=1)):
                for _, cluster in clstrs:
                    item_to_clusters[cluster].append(i + rng)

        self.clusters = []
        cluster_keeps = []
        for k, clust_sel in enumerate(clusters_selection):
            clustr = item_to_clusters[k]
            if len(clustr) > 0:
                mtx = similarity_type(self.records_features[clustr],
                                      self.records_data[clustr])
                self.clusters.append(mtx)
                cluster_keeps.append(clust_sel)

        self.root = similarity_type(cluster_keeps,
                                    list(range(len(cluster_keeps))))


    def search(self, records_features, k=1, min_threshold=0.95, 
               max_threshold=1.01, k_clusters=1, return_metric=True):
        """Find the closest item(s) for each feature_list in.

        Args:
            features_list: A list where each element is a list of features
                to query.
            k: Return the k closest results.
            min_threshold: Return items only at or above the threshold.
            max_threshold: Return items only at or below the threshold.
            k_clusters: number of clusters to search. This increases recall at
                the cost of some speed.
            return_metric: Return metric values? Metric can be a similarity
                value [0, 1] where 1 indicates similar (cosine similarity). 
                Metric can also be a distance measure (euclidean, hamming).

        Returns:
            For each element in features_list, return the k-nearest items
            and their similarity clores
            [[(score1_1, item1_1), ..., (score1_k, item1_k)],
             [(score2_1, item2_1), ..., (score2_k, item2_k)], ...]

             Note: if return_metric is False then only items are returned
                and not as a tuple.
        """
        # could make this recursive at the cost of recall accuracy
        # should batch requests to clusters to make this more efficent
        ret = []
        nearest = self.root.nearest_search(records_features, k=k_clusters)

        for i, nearest_clusters in enumerate(nearest):
            curr_ret = []

            for _, cluster in nearest_clusters:

                cluster_items = self.clusters[cluster].\
                        nearest_search([records_features[i]], k=k,
                                       min_threshold=min_threshold,
                                       max_threshold=max_threshold)

                for elements in cluster_items:
                    if len(elements) > 0:
                        if return_metric:
                            curr_ret.extend(elements)
                        else:
                            curr_ret.extend(elements)
            ret.append(k_best(curr_ret, k, return_metric, 
                              self.root.is_similarity))
        return ret
