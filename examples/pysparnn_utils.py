# Copyright 2016-present, Facebook, Inc.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE-examples file in the root directory of this source tree.
import numpy as np

# code that will measure query time and recall
def recall(query, full_set):
    ret =  []
    for r_items, t_items in zip(query, full_set):
        result = 0.0
        for r in np.unique(r_items):
            result += 1 if r in t_items else 0
        ret.append(result / len(t_items))
    return np.array(ret)