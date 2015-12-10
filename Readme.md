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

## Usage
To run the algorithm type

```shell
python cluster.py -d [data] -l [label_index]
```

where `[data]` is the path to your data file (comma delimited), and `[label_index]` is the index of the column
in the data file that contains the actual labels (usually 0). Results are stored in a folder named
 `results` with a text file corresponding to `[data]` with the following information:
 error on our algorithm together with error on some standard clustering algorithms -
**single linkage**, **average linkage**, **complete linkage** and **Ward's method**.
