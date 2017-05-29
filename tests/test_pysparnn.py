# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Test pysparn search"""

import unittest
import pysparnn.cluster_index as ci
import numpy as np
from scipy.sparse import csr_matrix
from pysparnn.matrix_distance import SlowEuclideanDistance
from pysparnn.matrix_distance import UnitCosineDistance
from pysparnn.matrix_distance import DenseCosineDistance
from sklearn.feature_extraction import DictVectorizer

class PysparnnTest(unittest.TestCase):
    """End to end tests for pysparnn"""
    def setUp(self):
        np.random.seed(1)

    def test_remove_duplicates(self):
        """Do a quick basic test for index/search functionality"""
        data = [
            'hello world',
            'hello world',
            'oh hello there',
            'oh hello there',
            'oh hello there',
            'Play it',
            'Play it again Sam',
        ]

        features = [dict([(x, 1) for x in f.split()]) for f in data]
        features = DictVectorizer().fit_transform(features)
        dist = UnitCosineDistance(features, data)
        
        self.assertEqual(dist.matrix.shape[0], 7)

        dist.remove_near_duplicates()

        self.assertEqual(dist.matrix.shape[0], 4)

    def test_cosine(self):
        """Do a quick basic test for index/search functionality"""
        data = [
            'hello world',
            'oh hello there',
            'Play it',
            'Play it again Sam',
        ]

        features = [dict([(x, 1) for x in f.split()]) for f in data]
        features = DictVectorizer().fit_transform(features)

        cluster_index = ci.ClusterIndex(features, data)

        ret = cluster_index.search(features, k=1, k_clusters=1,
                                   return_distance=False)
        self.assertEqual([[d] for d in data], ret)

    def test_dense_array(self):
        """Do a quick basic test for index/search functionality"""
        data = [
            'hello world',
            'oh hello there',
            'Play it',
            'Play it again Sam',
        ]

        features = [dict([(x, 1) for x in f.split()]) for f in data]
        features = DictVectorizer().fit_transform(features)
        features = features.toarray()
        cluster_index = ci.ClusterIndex(features, data)

        ret = cluster_index.search(features, k=1, k_clusters=1,
                                   return_distance=False)
        self.assertEqual([[d] for d in data], ret)

    def test_dense_matrix(self):
        """Do a quick basic test for index/search functionality"""
        data = [
            'hello world',
            'oh hello there',
            'Play it',
            'Play it again Sam',
        ]

        features = [dict([(x, 1) for x in f.split()]) for f in data]
        features = DictVectorizer().fit_transform(features)
        features = features.toarray()
        cluster_index = ci.ClusterIndex(features, data, DenseCosineDistance)

        ret = cluster_index.search(features, k=1, k_clusters=1,
                                   return_distance=False)
        self.assertEqual([[d] for d in data], ret)

    def test_euclidean(self):
        """Do a quick basic test for index/search functionality"""
        data = [
            'hello world',
            'oh hello there',
            'Play it',
            'Play it again Sam',
        ]

        features = [dict([(x, 1) for x in f.split()]) for f in data]
        features = DictVectorizer().fit_transform(features)
        features = features.toarray()
        cluster_index = ci.ClusterIndex(features, data, SlowEuclideanDistance)

        ret = cluster_index.search(features, k=1, k_clusters=1,
                                   return_distance=False)
        self.assertEqual([[d] for d in data], ret)



    def test_levels(self):
        """Test multiple level indexes"""
        features = np.random.binomial(1, 0.01, size=(1000, 20000))
        features = csr_matrix(features)

        # build the search index!
        data_to_return = np.array(list(range(1000)), dtype=int)

        # matrix size smaller - this forces the index to have multiple levels
        cluster_index = ci.ClusterIndex(features, data_to_return,
                                        matrix_size=10)

        ret =  cluster_index.search(features[0:10], k=1, k_clusters=1,
                                    return_distance=False)
        self.assertEqual([[x] for x in data_to_return[:10]], ret)

    def test_levels_multiindex(self):
        """Test multiple level indexes"""
        features = np.random.binomial(1, 0.01, size=(1000, 20000))
        features = csr_matrix(features)

        # build the search index!
        data_to_return = np.array(list(range(1000)), dtype=int)

        # matrix size smaller - this forces the index to have multiple levels
        cluster_index = ci.MultiClusterIndex(features, data_to_return,
                                             matrix_size=10)

        ret =  cluster_index.search(features[0:10], k=1, k_clusters=1,
                                    return_distance=False)
        self.assertEqual([[x] for x in data_to_return[:10]], ret)

    def test_large_k(self):
        """Test multiple level indexes"""
        features = np.random.binomial(1, 0.01, size=(1000, 20000))
        features = csr_matrix(features)

        # build the search index!
        data_to_return = np.array(list(range(1000)), dtype=int)

        # matrix size smaller - this forces the index to have multiple levels
        cluster_index = ci.MultiClusterIndex(features, data_to_return,
                                             matrix_size=10)

        ret =  cluster_index.search(features[0], k=100, k_clusters=1,
                                    return_distance=False)
        self.assertEqual(100, len(ret[0]))
