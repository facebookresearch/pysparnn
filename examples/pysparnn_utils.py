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
from sklearn.neighbors import LSHForest, NearestNeighbors 
from sklearn.feature_extraction import DictVectorizer
import pysparnn

class PySparNNTextSearch:
    def __init__(self, docs, k, matrix_size=None):
        self.dv = DictVectorizer()
        self.k = k
        datas = np.array(range(len(docs)))
        dicts = []
        for d in docs:
            dicts.append(dict([(w, 1) for w in d]))
        self.dv.fit(dicts)
        features = csr_matrix(self.dv.transform(dicts), dtype=int)
        self.cp = pysparnn.ClusterIndex(features, datas, matrix_size=matrix_size)
        
    def search(self, docs):
        dicts = []
        for d in docs:
            dicts.append(dict([(w, 1) for w in d]))
        features = csr_matrix(self.dv.transform(dicts), dtype=int)
        return self.cp.search(features, k=self.k, k_clusters=1, return_distance=False)

class LSHForestTextSearch:
    def __init__(self, docs, k):
        self.lshf = LSHForest(n_estimators=1, n_candidates=1,
                     n_neighbors=k)
        self.dv = DictVectorizer()
        self.k = k
        
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
        return self.lshf.kneighbors(features, return_distance=False, n_neighbors=self.k)    
    
class KNNTextSearch:
    def __init__(self, docs, k):
        self.knn = NearestNeighbors(n_neighbors=k)
        self.dv = DictVectorizer()
        self.k = k
        dicts = []
        for d in docs:
            dicts.append(dict([(w, 1) for w in d]))
        self.dv.fit(dicts)
        features = self.dv.transform(dicts)
        # floats are faster
        # features = csr_matrix(features, dtype=int)
        self.knn.fit(features)
        
    def search(self, docs):
        dicts = []
        for d in docs:
            dicts.append(dict([(w, 1) for w in d]))
        features = self.dv.transform(dicts)
        # floats are faster
        # features = csr_matrix(features, dtype=int)
        return self.knn.kneighbors(features, return_distance=False, n_neighbors=self.k)        

    
class PySparNNSearch:
    def __init__(self, features, k, matrix_size=None):
        self.k = k
        datas = np.array(range(len(features)))
        features = csr_matrix(features)
        self.cp = pysparnn.ClusterIndex(features, datas, matrix_size=matrix_size)
        
    def search(self, features):
        features = csr_matrix(features)
        return self.cp.search(features, k=self.k, k_clusters=1, return_distance=False)

class LSHForestSearch:
    def __init__(self, features, k):
        self.lshf = LSHForest(n_estimators=1, n_candidates=1,
                     n_neighbors=k)
        self.k = k
        
        self.lshf.fit(features)
        
    def search(self, features):
        
        return self.lshf.kneighbors(features, return_distance=False, n_neighbors=self.k)    
    
class KNNSearch:
    def __init__(self, features, k):
        self.knn = NearestNeighbors(n_neighbors=k)
        self.knn.fit(features)
        self.k = k
        
    def search(self, features):
        return self.knn.kneighbors(features, return_distance=False, n_neighbors=self.k)            
    
# code that will measure query time and recall
def recall(result, truth):
    ret =  []
    for r_items, t_items in zip(result, truth):
        result = 0.0
        for r in r_items:
            result += 1 if r in t_items else 0
        ret.append(result / len(t_items))
    return np.array(ret)

def time_it(search_index, docs, query_index, answers):
    query_docs = []
    query_answers = []    
    for i in query_index:
        query_docs.append(docs[i])
        query_answers.append(answers[i])
    
    # time how long the query takes
    t0 = time.time()
    neighbors = search_index.search(query_docs)
    delta = time.time() - t0

    return delta, recall(neighbors, query_answers).mean()

def knn_benchmark(search_index, docs, answers, n_trials=1000, docs_per_query=50):
    # Bootstrap-ish code to measure the time and accuracy
    times = []
    recalls = []
    for i in range(n_trials):
        query_index = random.sample(range(len(docs)), docs_per_query)
        time, recall = time_it(search_index, docs, query_index, answers)
        time = time / docs_per_query
        times.append(time)
        recalls.append(recall)
    return np.median(times), np.median(recalls)    