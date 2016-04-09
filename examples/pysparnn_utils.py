import time
import random
import numpy as np

def accuracy(result, truth):
    ret =  []
    for r, t in zip(result, truth):
        ret.append(1 if t in r else 0)
    return np.array(ret)

def time_it(search_index, docs, query_index):
    # time how long the query takes
    t0 = time.time()
    neighbors = search_index.search(docs[query_index])
    delta = time.time() - t0

    return delta, accuracy(neighbors, query_index).mean()

def time_it_n(search_index, docs, n=500, k_docs=20):
    # a rough bootstrap
    times = []
    accuracys = []
    for i in range(n):
        query_index = random.sample(range(len(docs)), k_docs)
        time, accuracy = time_it(search_index, docs, query_index)
        time = time / k_docs
        times.append(time)
        accuracys.append(accuracy)
    return np.median(times), np.median(accuracys)