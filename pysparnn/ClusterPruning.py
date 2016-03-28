# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant 
# of patent rights can be found in the PATENTS file in the same directory.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
import math
import numpy as np
import random
import scipy.sparse

def k_best(l, k, similarity): 
    """Get the k-best tuples by similarity.
    Args:
        l: List of tuples.
        k: Number of tuples to return.
        similarity: Boolean value indicating if similarity values should be
            returned.
    Returns:
        The K-best tuples (similarity, value) by similarity score.
    """
    l = sorted(l, key=lambda x: x[0], reverse=True)[:k]
    if similarity:
        return l
    else:
        return [x[1] for x in l]

class MatrixCluster(object):
    """A sparse matrix representation out of features."""

    def __init__(self, records_features, records_data):
        """
        Args: 
            records_features: List of features in the format of 
               {feature_name1 -> value1, feature_name2->value2, ...}.
            records_data: Data to return when a doc is matched. Index of
                corresponds to records_features.
        """
        self.dimension = {}
        self.inverse_dimension = {}
        self.matrix = self._create_matrix(records_features, 
                expand_dimension=True)
        self.records_features = np.array(records_features)
        self.records_data = np.array(records_data)


    def _create_matrix(self, records_features, expand_dimension=False):
        """Create a sparse matrix out of a set of features.
        Args: 
            records_features: List of features in the format of 
               {feature_name1 -> value1, feature_name2->value2, ...}.
            records_data: Data to return when a doc is matched. Index of
                corresponds to records_features.
            expand_dimension: Should the dimension of the space be expanded?
                True on initialization. False on search.
        """
        indptr = [0]
        indices = []
        data = []
        # could force records_features to be a list of (int, float) instead of 
        #  ageneric dict (that can take strings)
        for features in records_features:
            for feature, value in features.iteritems():
                if expand_dimension or feature in self.dimension:
                    index = self.dimension.setdefault(feature, 
                            len(self.dimension))
                    self.inverse_dimension[index] = feature
                    indices.append(index)
                    data.append(self._transform_value(value))
            indptr.append(len(indices))

        shape = (len(records_features), len(self.dimension))
        return scipy.sparse.csr_matrix((data, indices, indptr), dtype=float,
            shape=shape)    
    

    def nearest_search(self, features, k=1, threshold=0.0):
        """Find the closest item(s) for each feature_list in  

        Args: 
            features_list: A list where each element is a list of features
                to query.
            k: Return the k closest results.
            threshold: Return items only above the threshold.

        Returns:
            For each element in features_list, return the k-nearest items
            and their similarity clores 
            [[(score1_1, item1_1), ..., (score1_k, item1_k)], 
             [(score2_1, item2_1), ..., (score2_k, item2_k)], ...]
        """
        a = self._create_matrix(features)
        sim_matrix = self._similarity(a).toarray()
        sim_filter = sim_matrix >= threshold

        ret = []
        for i in range(sim_matrix.shape[0]):
            # these arrays are the length of the sqrt(index)
            # replacing the for loop by matrix ops could speed things up
            
            index = sim_filter[i]
            scores = sim_matrix[i][index]
            records = self.records_data[index]
            arg_index = np.argsort(scores)[-k:]
            
            curr_ret = zip(scores[arg_index], records[arg_index])
            
            ret.append(curr_ret)
        
        return ret    
        
#class UnitVecCosineMatrixCluster(MatrixCluster):
#    def __init__(self, records_features, records_data):
#        super(UnitVecCosineMatrixCluster, self).__init__(records_features, 
#                records_data)
#        
#        # we inforce 1 hot encodeing. this means that all our values are
#        # 0 or 1
#        # since 1^2 == 1, we can do a sum shortcut instad of sum of squares
#        # this is much faster and more memory efficent
#        self.matrix_root_sum_square = \
#                np.sqrt(np.asarray(self.matrix.sum(axis=1)).reshape(-1))
#    
#    def _transform_value(self, v):
#        return 1
#    
#    def _similarity(self, a):
#        """Vectorised cosine similarity"""
#        dprod = a.dot(self.matrix.transpose()) * 1.0
#
#        a_root_sum_square = np.asarray(a.sum(axis=1)).reshape(-1)
#        a_root_sum_square = a_root_sum_square.reshape(len(a_root_sum_square), 1)
#        a_root_sum_square = np.sqrt(a_root_sum_square)
#        
#        magnitude = 1.0 / (a_root_sum_square * self.matrix_root_sum_square)
#
#        return dprod.multiply(magnitude)

class CosineSimilarity(MatrixCluster):
    def __init__(self, records_features, records_data):
        super(CosineSimilarity, self).__init__(records_features, records_data)
        
        m_c = self.matrix.copy()
        m_c.data **= 2
        self.matrix_root_sum_square = \
                np.sqrt(np.asarray(m_c.sum(axis=1)).reshape(-1))
    
    def _transform_value(self, v):
        return v
    
    def _similarity(self, a):
        """Vectorised cosine similarity"""
        dprod = a.dot(self.matrix.transpose()) * 1.0

        a_c = a.copy()
        a_c.data **= 2
        a_root_sum_square = np.asarray(a_c.sum(axis=1)).reshape(-1)
        a_root_sum_square = \
                a_root_sum_square.reshape(len(a_root_sum_square), 1)
        a_root_sum_square = np.sqrt(a_root_sum_square)
        
        magnitude = 1.0 / (a_root_sum_square * self.matrix_root_sum_square)

        return dprod.multiply(magnitude)

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
            similarity_type=CosineSimilarity):
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
        for k in range(len(clusters_selection)):
            v = item_to_clusters[k]
            if len(v) > 0:
                mtx = similarity_type(self.records_features[v], 
                        self.records_data[v])
                self.clusters.append(mtx)
                cluster_keeps.append(clusters_selection[k])

        self.root = similarity_type(cluster_keeps,
                list(range(len(cluster_keeps))))


    # TODO: I think i can save a little time by batching the searches together
    #  or creating one huge matrix
    # TODO: Cut down index construction time
    # TODO: Speed comparison tests
    def search(self, records_features, k=1, threshold=0.95, k_clusters=1, 
            return_similarity=True):
        """Find the closest item(s) for each feature_list in.

        Args: 
            features_list: A list where each element is a list of features
                to query.
            k: Return the k closest results.
            threshold: Return items only above the threshold.
            k_clusters: number of clusters to search. This increases recall at
                the cost of some speed.
            return_similarity: Return similarity values?

        Returns:
            For each element in features_list, return the k-nearest items
            and their similarity clores 
            [[(score1_1, item1_1), ..., (score1_k, item1_k)], 
             [(score2_1, item2_1), ..., (score2_k, item2_k)], ...]

             Note: if return_similarity is False then only items are returned
                and not as a tuple.
        """
        # could make this recursive at the cost of recall accuracy
        # should batch requests to clusters to make this more efficent
        ret = []
        nearest = self.root.nearest_search(records_features, k=k_clusters)
        
        # TODO: np.array-ify - this loop can be replaced by array concats
        for i, nearest_clusters in enumerate(nearest):
            curr_ret = []
            
            for score, cluster in nearest_clusters:
                
                cluster_items = self.clusters[cluster].nearest_search(
                        [records_features[i]], k=k, threshold=threshold)
                
                for elements in cluster_items:
                    if len(elements) > 0:
                        if return_similarity:
                            curr_ret.extend(elements)
                        else:
                            curr_ret.extend(elements)
            ret.append(k_best(curr_ret, k, return_similarity))
        return ret
