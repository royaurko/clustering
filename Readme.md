# A noise tolerant stability-based algorithm for similarity clustering 

This is a Python 3 implementation of a simple thresholding algorithm for clustering under a pairwise
 similarity measure. The algorithm correctly recovers the **target** clustering if the similarity
 function k satisfies the following stability property

```
k(A, X) > k(A, Y) 
```

for every triple of sets A, X and Y such that A and X belong to the same cluster in the target cluster
while the set Y is a subset of points in some other cluster. Intuitively this captures the notion that sets of points
are more similar to points in their own cluster than to points in other clusters. The algorithm is noise 
stable in the sense that it only requires sets A, X and Y above a certain size limit (proportional to n) to
satisfy this property. The present implementation is not super optimized, however it is largely parallelized.

Authors: Aurko Roy, Sadra Yazdanbod and Daniel Zink

## Non standard dependencies
 - [munkres](https://pypi.python.org/pypi/munkres/)
 

## Usage
To run the algorithm type

```shell
python3 cluster.py -d [data] -l [label_index] -m [metric] -o [out_file] -n [num_workers]
```

* `[data]` is the path to the data file (comma delimited)
* `[label_index]` is the index of the column that contains the labels (default 0)
* `[metric]` is one of `{avg, max, min}`
* `[out_file]` is the pickle file to write the results into (default `results.pkl`)
* `[num_workers]` is the number of workers for parallelization (default 1)

The results pickle file contains a `dict` storing the 
error on this clustering algorithm (with the appropriate metric) together 
with the error on some standard clustering algorithms -
**single linkage**, **average linkage**, **complete linkage** and **Ward's method**.
