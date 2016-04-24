# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import time
import random
import numpy as np
import time
from scipy.sparse import csr_matrix
from sklearn.datasets import fetch_20newsgroups
from sklearn.neighbors import LSHForest
from sklearn.feature_extraction import DictVectorizer
import pysparnn

class PySparNNTextSearch:
    def __init__(self, docs, datas, matrix_size=2000):
        
        self.dv = DictVectorizer()
        dicts = []
        for d in docs:
            dicts.append(dict([(w, 1) for w in d]))
        self.dv.fit(dicts)
        features = csr_matrix(self.dv.transform(dicts), dtype=int)
        self.cp = pysparnn.ClusterIndex(features, datas, pysparnn.matrix_distance.UnitCosineDistance, matrix_size=matrix_size)
        
    def search(self, docs, k=1, min_distance=None, max_distance=None, k_clusters=1, return_distance=False):
        dicts = []
        for d in docs:
            dicts.append(dict([(w, 1) for w in d]))
        features = csr_matrix(self.dv.transform(dicts), dtype=int)
        return self.cp.search(features, k=k, min_distance=min_distance, max_distance=max_distance, 
                              k_clusters=k_clusters, return_distance=return_distance)

class LSHForestSearch:
    def __init__(self, docs):
        self.lshf = LSHForest(n_estimators=1, n_candidates=1,
                     n_neighbors=1)
        self.dv = DictVectorizer()
        dicts = []
        for d in docs:
            dicts.append(dict([(w, 1) for w in d]))
        self.dv.fit(dicts)
        features = self.dv.transform(dicts)
        # floats are faster
        # features = csr_matrix(features, dtype=int)
        self.lshf.fit(features)
        
    def search(self, docs):
        dicts = []
        for d in docs:
            dicts.append(dict([(w, 1) for w in d]))
        features = self.dv.transform(dicts)
        # floats are faster
        # features = csr_matrix(features, dtype=int)
        return self.lshf.kneighbors(features, return_distance=False)    
    
# code that will measure query time and accuracy
def accuracy(result, truth):
    ret =  []
    for r, t in zip(result, truth):
        ret.append(1 if t in r else 0)
    return np.array(ret)

def time_it(search_index, docs, query_index):
    query_docs = []
    for i in query_index:
        query_docs.append(docs[i])
    
    # time how long the query takes
    t0 = time.time()
    neighbors = search_index.search(query_docs)
    delta = time.time() - t0

    return delta, accuracy(neighbors, query_index).mean()

def identity_benchmark(search_index, docs, n_trials=1000, docs_per_query=50):
    # Bootstrap-ish code to measure the time and accuracy
    times = []
    accuracys = []
    for i in range(n_trials):
        query_index = random.sample(range(len(docs)), docs_per_query)
        time, accuracy = time_it(search_index, docs, query_index)
        time = time / docs_per_query
        times.append(time)
        accuracys.append(accuracy)
    return np.median(times), np.median(accuracys)    