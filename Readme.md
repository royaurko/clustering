# A thresholding algorithm for clustering

This is a (naive) Python implementation of a simple thresholding algorithm for clustering under a pairwise
 similarity measure. The algorithm correctly recovers the **target** clustering if the similarity
 function k satisfies the following stability property

```
k(A, X) > k(A, Y) 
```

for every triple of sets A, X and Y such that A and X belong to the same cluster in the target cluster
while the set Y is a subset of points in some other cluster. Intuitively this captures the notion that sets of points
 are more similar to points in their own cluster than to points in other clusters. The algorithm is noise 
stable in the sense that it only requires sets A, X and Y above a certain size limit (proportional to n) to
 satisfy this property. The algorithm can be implemented to run in time O(n^4), but the present implementation
 is pretty naive and can be optimized further. I will put up a link to the manuscript later.

Authors (manuscript): Aurko Roy, Sadra Yazdanbod and Daniel Zink

## Dependencies
 - [numpy](http://www.numpy.org/)
 - [scipy](http://www.scipy.org/)
 - [munkres](https://pypi.python.org/pypi/munkres/)

## Installation
The following will install the algorithm as a python module

```shell
pip install "git+https://github.com/royaurko/threshold-clustering#egg=threshold-clustering"
```

## Usage
To run the actual algorithm type the following in a python interpreter

```python
from tcluster import cluster
cluster.test(data, target, k, e)
```
where `data` the numpy array representing the data, `target` is the target labels, `k` is the number of clusters in your target
and `e`is a number between 0 and 1 which denotes the minimum fraction of points contained in a single cluster in your target. If
the labels are evenly balanced then you may set `e` = `1/k`.

The *cluster.test()* function computes the error of our algorithm and compares it to standard clustering algorithms -
**single linkage**, **average linkage**, **complete linkage** and **Ward's method**.
