# PySparNN
Approximate Nearest Neighbor Search for Sparse Data in Python! This library is well suited to finding nearest neighbors in sparse, high dimensional spaces (like text documents). 

Out of the box, PySparNN supports Cosine Similarity.

PySparNN benefits:
 * Designed to be efficent on sparse data (both on memory and cpu).
 * Implemented leveraging existing python libraries (scipy & numpy).
 * Easily extended with other metrics: Manhattan, Eculidian, Jaccard, etc.
 * (experimental) Min, Max similairty thresholds can be set at query time (not index time). I.e. return the k closest items between 0.9 and 0.8 cosine similarity from an input point.

If your data is NOT SPARSE - please consider [annoy](https://github.com/spotify/annoy). Annoy uses a similar-ish method and I am a big fan of it. As of this writing, annoy performs ~8x faster on their introductory example. 
General rule of thumb - annoy performs better if you can get your data to fit into memory (as a dense vector).


The most comparable library to PySparNN is scikit-learn's LSHForrest module. As of this writing, PySparNN is 1.68x faster on the 20newsgroups dataset. A more thurough benchmarking on sparse data is desired. [Here is the comparison.](https://github.com/facebookresearch/pysparnn/blob/master/sparse_search_comparison.ipynb)

Notes:
* A future update may allow incremental insertions.

## Example Usage
### Simple Example
```
import pysparnn as snn

import numpy as np
from scipy.sparse import csr_matrix

features = np.random.binomial(1, 0.01, size=(1000, 20000))
features = csr_matrix(features)

# build the search index!
data_to_return = range(1000)
cp = snn.ClusterIndex(features, data_to_return)

cp.search(features[:5], min_threshold=0.50, k=1, return_metric=False)
>> [[0], [1], [2], [3], [4]]
```
### Text Example
```
import pysparnn as snn

from sklearn.feature_extraction import DictVectorizer

data = [
    'hello world',
    'oh hello there',
    'Play it',
    'Play it again Sam',
]    

# build a feature representation for each sentence
def scentence2features(scentence):
    features = dict()
    for word in scentence.split():
        features[word] = 1
    return features

features_list = []
for sentence in data:
    features_list.append(scentence2features(sentence))

dv = DictVectorizer()
dv.fit(features_list)

# build the search index!
cp = snn.ClusterIndex(dv.transform(features_list), data)

# search the index with a sparse matrix
search_items = [
    scentence2features('oh there'),
    scentence2features('Play it again Frank')
]
search_items = dv.transform(search_items)

cp.search(search_items, min_threshold=0.50, k=1, k_clusters=2, return_metric=False)
>> [['oh hello there'], ['Play it again Sam']]

```

## Requirements
PySparNN requires numpy and scipy. Tested with numpy 1.10.4 and scipy 0.17.0.

## How PySparNN works
Searching for a document in an collection of K documents is naievely O(K) (assuming documents are constant sized). 

However! we can create a tree structure where the first level is O(sqrt(K)) and each of the leaves are also O(sqrt(K)) - on average.

We randomly pick sqrt(K) candidate items to be in the top level. Then -- each document in the full list of K documents is assigned to the closest candidate in the top level.

This breaks up one O(K) search into two O(sqrt(K)) searches which is much much faster when K is big!

## Further Information
http://nlp.stanford.edu/IR-book/html/htmledition/cluster-pruning-1.html

See the CONTRIBUTING file for how to help out.

## License
PySparNN is BSD-licensed. We also provide an additional patent grant.
