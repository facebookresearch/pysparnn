# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Defines a distance search structure"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import numpy as np
import scipy.sparse
import scipy.spatial.distance

class MatrixMetricSearch(object):
    """A sparse matrix representation out of features."""
    __metaclass__ = abc.ABCMeta

    def __init__(self, sparse_features, records_data):
        """
        Args:
            sparse_features: A csr_matrix with rows that represent records
                (corresponding to the elements in records_data) and columns
                that describe a point in space for each row.
            records_data: Data to return when a doc is matched. Index of
                corresponds to sparse_features.
        """
        self.matrix = sparse_features
        self.records_data = np.array(records_data)

    def get_feature_matrix(self):
        return self.matrix

    def get_records(self):
        return self.records_data

    @abc.abstractmethod
    def _transform_value(self, val):
        """
        Args:
            val: A numeric value to be (potentially transformed).
        Returns:
            The transformed numeric value.
        """
        return

    @abc.abstractmethod
    def _distance(self, a_matrix):
        """
        Args:
            a_matrix: A csr_matrix with rows that represent records
                to search against.
            records_data: Data to return when a doc is matched. Index of
                corresponds to sparse_features.
        Returns:
            A dense array representing distance.
        """
        return

    def nearest_search(self, sparse_features, k=1, max_distance=None):
        """Find the closest item(s) for each set of features in features_list.

        Args:
            sparse_features: A csr_matrix with rows that represent records
                (corresponding to the elements in records_data) and columns
                that describe a point in space for each row.
            k: Return the k closest results.
            max_distance: Return items at most max_distance from the query
                point.

        Returns:
            For each element in features_list, return the k-nearest items
            and their distance scores
            [[(score1_1, item1_1), ..., (score1_k, item1_k)],
             [(score2_1, item2_1), ..., (score2_k, item2_k)], ...]
        """

        dist_matrix = self._distance(sparse_features)

        if max_distance is None:
            max_distance = float("inf")

        dist_filter = (dist_matrix <= max_distance)

        ret = []
        for i in range(dist_matrix.shape[0]):
            # these arrays are the length of the sqrt(index)
            # replacing the for loop by matrix ops could speed things up

            index = dist_filter[i]
            scores = dist_matrix[i][index]
            records = self.records_data[index]

            if scores.sum() < 0.0001 and len(scores) > 0:
                # they are all practically the same
                # we have to do this to prevent infinite recursion
                # TODO: would love an alternative solution, this is a critical loop
                arg_index = np.random.choice(len(scores), k, replace=False)
            else:
                arg_index = np.argsort(scores)[:k]

            curr_ret = zip(scores[arg_index], records[arg_index])

            ret.append(curr_ret)

        return ret

class CosineDistance(MatrixMetricSearch):
    """A matrix that implements cosine distance search against it.

    cosine_distance = 1 - cosine_similarity

    Note: We want items that are more similar to be closer to zero so we are
    going to instead return 1 - cosine_similarity. We do this so similarity
    and distance metrics can be treated the same way.
    """

    def __init__(self, sparse_features, records_data):
        super(CosineDistance, self).__init__(sparse_features, records_data)

        m_c = self.matrix.copy()
        m_c.data **= 2
        self.matrix_root_sum_square = \
                np.sqrt(np.asarray(m_c.sum(axis=1)).reshape(-1))

    def _transform_value(self, v):
        return v

    def _distance(self, a_matrix):
        """Vectorised cosine distance"""
        # what is the implmentation of transpose? can i change the order?
        dprod = self.matrix.dot(a_matrix.transpose()).transpose() * 1.0

        a_c = a_matrix.copy()
        a_c.data **= 2
        a_root_sum_square = np.asarray(a_c.sum(axis=1)).reshape(-1)
        a_root_sum_square = \
                a_root_sum_square.reshape(len(a_root_sum_square), 1)
        a_root_sum_square = np.sqrt(a_root_sum_square)

        magnitude = 1.0 / (a_root_sum_square * self.matrix_root_sum_square)

        return 1 - dprod.multiply(magnitude).toarray()

class UnitCosineDistance(MatrixMetricSearch):
    """A matrix that implements cosine distance search against it.

    cosine_distance = 1 - cosine_similarity

    Note: We want items that are more similar to be closer to zero so we are
    going to instead return 1 - cosine_similarity. We do this so similarity
    and distance metrics can be treated the same way.

    Assumes unit-vectors and takes some shortucts:
      * Uses integers instead of floats
      * 1**2 == 1 so that operation can be skipped
    """

    def __init__(self, sparse_features, records_data):
        super(UnitCosineDistance, self).__init__(sparse_features, records_data)
        self.matrix_root_sum_square = \
                np.sqrt(np.asarray(self.matrix.sum(axis=1)).reshape(-1))

    def _transform_value(self, v):
        return 1

    def _distance(self, a_matrix):
        """Vectorised cosine distance"""
        # what is the implmentation of transpose? can i change the order?
        dprod = self.matrix.dot(a_matrix.transpose()).transpose() * 1.0

        a_root_sum_square = np.asarray(a_matrix.sum(axis=1)).reshape(-1)
        a_root_sum_square = \
                a_root_sum_square.reshape(len(a_root_sum_square), 1)
        a_root_sum_square = np.sqrt(a_root_sum_square)

        magnitude = 1.0 / (a_root_sum_square * self.matrix_root_sum_square)

        return 1 - dprod.multiply(magnitude).toarray()

class SlowEuclideanDistance(MatrixMetricSearch):
    """A matrix that implements euclidean distance search against it.
    WARNING: This is not optimized.
    """

    def __init__(self, sparse_features, records_data):
        super(SlowEuclideanDistance, self).__init__(sparse_features,
                                                    records_data)
        self.matrix = self.matrix.toarray()

    def _transform_value(self, v):
        return v

    def _distance(self, a_matrix):
        """Euclidean distance"""

        return scipy.spatial.distance.cdist(a_matrix.toarray(), self.matrix,
                                            'euclidean')
