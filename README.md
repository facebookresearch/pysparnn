# PySparNN
Approximate Nearest Neighbor Search for Sparse Data in Python! This library is well suited to finding nearest neighbors in sparse, high dimensional spaces (like text documents). 

Out of the box, PySparNN supports Cosine Distance (i.e. 1 - cosine_similarity).

PySparNN benefits:
 * Designed to be efficient on sparse data (memory & cpu).
 * Implemented leveraging existing python libraries (scipy & numpy).
 * Easily extended with other metrics: Manhattan, Euclidian, Jaccard, etc.
 * Max distance thresholds can be set at query time (not index time). I.e. return the k closest items no more than max_distance from the query point.
 * Supports incremental insertion of elements 

If your data is NOT SPARSE - please consider [annoy](https://github.com/spotify/annoy). Annoy uses a similar-ish method and I am a big fan of it. As of this writing, annoy performs ~8x faster on their introductory example. 
General rule of thumb - annoy performs better if you can get your data to fit into memory (as a dense vector).

The most comparable library to PySparNN is scikit-learn's LSHForrest module. As of this writing, PySparNN is ~2x faster on the 20newsgroups dataset. A more robust benchmarking on sparse data is desired. [Here is the comparison.](https://github.com/facebookresearch/pysparnn/blob/master/examples/sparse_search_comparison.ipynb)

## Example Usage
### Simple Example
```python
import pysparnn as snn

import numpy as np
from scipy.sparse import csr_matrix

features = np.random.binomial(1, 0.01, size=(1000, 20000))
features = csr_matrix(features)

# build the search index!
data_to_return = range(1000)
cp = snn.ClusterIndex(features, data_to_return)

cp.search(features[:5], k=1, return_distance=False)
>> [[0], [1], [2], [3], [4]]
```
### Text Example
```python
import pysparnn as snn

from sklearn.feature_extraction.text import TfidfVectorizer

data = [
    'hello world',
    'oh hello there',
    'Play it',
    'Play it again Sam',
]    

tv = TfidfVectorizer()
tv.fit(data)

features_vec = tv.transform(data)

# build the search index!
cp = snn.ClusterIndex(features_vec, data)

# search the index with a sparse matrix
search_data = [
    'oh there',
    'Play it again Frank'
]

search_features_vec = tv.transform(search_data)

cp.search(search_features_vec, k=1, k_clusters=2, return_distance=False)
>> [['oh hello there'], ['Play it again Sam']]

```

## Requirements
PySparNN requires numpy and scipy. Tested with numpy 1.10.4 and scipy 0.17.0.

## Installation
```bash
# clone pysparnn
cd pysparnn 
pip install -r requirements.txt 
python setup.py install
```

## How PySparNN works
Searching for a document in an collection of K documents is naively O(K) (assuming documents are constant sized). 

However! we can create a tree structure where the first level is O(sqrt(K)) and each of the leaves are also O(sqrt(K)) - on average.

We randomly pick sqrt(K) candidate items to be in the top level. Then -- each document in the full list of K documents is assigned to the closest candidate in the top level.

This breaks up one O(K) search into two O(sqrt(K)) searches which is much much faster when K is big!

This generalizes to h levels. The runtime becomes:
    O(h * h_root(K))

**Note on min_distance thresholds** - Each document is assigned to the closest candidate cluster. When we set min_distance we will filter out clusters that don't meet that requirement without going into the individual clusters looking for matches. This means that we are likely to miss some good matches along the way since we wont investigate clusters that just miss the cutoff. A (planned) patch for this behavior would be to also search clusters that 'just' miss this cutoff. 

## Further Information
http://nlp.stanford.edu/IR-book/html/htmledition/cluster-pruning-1.html

See the CONTRIBUTING file for how to help out.

## License
PySparNN is BSD-licensed. We also provide an additional patent grant.
