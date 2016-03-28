# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Test pysparn search"""

import unittest
import pysparnn as snn

class PysparnnTest(unittest.TestCase):
    """End to end tests for pysparnn"""
    def test(self):
        """Do a quick basic test for index/search functionality"""
        data = [
            'hello world',
            'oh hello there',
            'Play it',
            'Play it again Sam',
        ]

        features = [dict([(x, 1) for x in f.split()]) for f in data]

        cluster_index = snn.ClusterIndex(features, data)

        ret = cluster_index .search(features, threshold=0.50, k=1,
                                    k_clusters=1, return_similarity=False)
        self.assertEqual([[d] for d in data], ret)
