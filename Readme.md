# A thresholding algorithm for clustering

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


Authors (manuscript): Aurko Roy, Sadra Yazdanbod and Daniel Zink

## Non standard dependencies
 - [munkres](https://pypi.python.org/pypi/munkres/)
 

## Usage
To run the algorithm type

```shell
python cluster.py -d [data] -l [label_index] -n [num_workers]
```

where `[data]` is the path to your data file (comma delimited), `[label_index]` is the index of the column
in the data file that contains the actual labels (default 0) and `[num_workers]` is the number of workers (cores)
you want to use for parallelization (default 1). Results are stored in a 
pickle file named `results.pkl` with the following information:
 error on this clustering algorithm together with error on some standard clustering algorithms -
**single linkage**, **average linkage**, **complete linkage** and **Ward's method**.
