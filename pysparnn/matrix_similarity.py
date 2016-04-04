# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Defines a similarity search structure for doing similarity search"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import numpy as np
import scipy.sparse

class MatrixSimilaritySearch(object):
    """A sparse matrix representation out of features."""
    __metaclass__ = abc.ABCMeta

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

    @abc.abstractmethod
    def _transform_value(self, val):
        return

    @abc.abstractmethod
    def _similarity(self, a_matrix):
        return

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

    def nearest_search(self, features_list, k=1, threshold=0.0):
        """Find the closest item(s) for each set of features in features_list.

        Args:
            features_list: A list where each element is a list of features
                to query.
            k: Return the k closest results.
            threshold: Return items only above the threshold.

        Returns:
            For each element in features_list, return the k-nearest items
            and their similarity scores
            [[(score1_1, item1_1), ..., (score1_k, item1_k)],
             [(score2_1, item2_1), ..., (score2_k, item2_k)], ...]
        """
        a_matrix = self._create_matrix(features_list)
        sim_matrix = self._similarity(a_matrix).toarray()
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

class CosineSimilarity(MatrixSimilaritySearch):
    """A matrix that implements cosine similarity search against it."""

    def __init__(self, records_features, records_data):
        super(CosineSimilarity, self).__init__(records_features, records_data)

        m_c = self.matrix.copy()
        m_c.data **= 2
        self.matrix_root_sum_square = \
                np.sqrt(np.asarray(m_c.sum(axis=1)).reshape(-1))

    def _transform_value(self, v):
        return v

    def _similarity(self, a_matrix):
        """Vectorised cosine similarity"""
        dprod = a_matrix.dot(self.matrix.transpose()) * 1.0

        a_c = a_matrix.copy()
        a_c.data **= 2
        a_root_sum_square = np.asarray(a_c.sum(axis=1)).reshape(-1)
        a_root_sum_square = \
                a_root_sum_square.reshape(len(a_root_sum_square), 1)
        a_root_sum_square = np.sqrt(a_root_sum_square)

        magnitude = 1.0 / (a_root_sum_square * self.matrix_root_sum_square)

        return dprod.multiply(magnitude)
