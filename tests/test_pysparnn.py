# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Test pysparn search"""

import unittest
import pysparnn.cluster_pruning as cp
from pysparnn.matrix_similarity import UnitCosineSimilarity
from pysparnn.matrix_similarity import EuclideanDistance

class PysparnnTest(unittest.TestCase):
    """End to end tests for pysparnn"""
    def test_cosine(self):
        """Do a quick basic test for index/search functionality"""
        data = [
            'hello world',
            'oh hello there',
            'Play it',
            'Play it again Sam',
        ]

        features = [dict([(x, 1) for x in f.split()]) for f in data]

        cluster_index = cp.ClusterIndex(features, data)

        ret = cluster_index.search(features, min_threshold=0.50, k=1,
                                    k_clusters=1, return_metric=False)
        self.assertEqual([[d] for d in data], ret)

    def test_cosine_unit(self):
        """Do a quick basic test for index/search functionality"""
        data = [
            'hello world',
            'oh hello there',
            'Play it',
            'Play it again Sam',
        ]

        features = [dict([(x, 1) for x in f.split()]) for f in data]

        cluster_index = cp.ClusterIndex(features, data, UnitCosineSimilarity)

        ret = cluster_index.search(features, min_threshold=0.50, k=1,
                                    k_clusters=1, return_metric=False)
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

        cluster_index = cp.ClusterIndex(features, data, EuclideanDistance)

        ret = cluster_index.search(features, min_threshold=0.0, 
                                   max_threshold=0.5, k=1, k_clusters=1, 
                                   return_metric=False)
        self.assertEqual([[d] for d in data], ret)
