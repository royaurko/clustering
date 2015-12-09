import scipy.cluster.hierarchy as hac
import numpy as np
from scipy.spatial.distance import pdist
from scipy.linalg import norm
from munkres import Munkres
import time
from joblib import Parallel, delayed
import multiprocessing
num_cpu = multiprocessing.cpu_count()


def error(cluster, tcluster):
    """ Compute error between cluster and target cluster
    :param cluster: proposed cluster
    :param tcluster: target cluster
    :return: error
    """
    k = len(set(tcluster))
    n = len(tcluster)
    C = list()
    T = list()
    for i in range(1, k+1):
        tmp = {j for j in range(n) if cluster[j] == i}
        C.append(tmp)
        tmp = {j for j in range(n) if tcluster[j] == i}
        T.append(tmp)
    M = list()
    for i in range(k):
        M.append([0]*k)
    testM = list()
    for i in range(k):
        testM.append([0]*k)
    for i in range(k):
        for j in range(k):
            M[i][j] = len(C[i].difference(T[j]))
            testM[i][j] = len(T[j].difference(C[i]))
    m = Munkres()
    indexes = m.compute(M)
    total = 0
    for row, col in indexes:
        value = M[row][col]
        total += value
    indexes2 = m.compute(testM)
    total2 = 0
    for row, col in indexes2:
        value = testM[row][col]
        total2 += value
    return float(total)/float(n)


def set_distances(X, S):
    """ Return the elements outside S sorted by distance to S
    :param X: Data matrix
    :param S: set of points
    :return: Elements outside S sorted by distance to S
    """
    n = len(X)
    s = len(S)
    dist = list()
    for i in range(n):
        if i not in S:
            d = 0.0
            for j in S:
                d += norm(X[i] - X[j])
            d /= s
            dist.append((i, d))
    dist = sorted(dist, key=lambda x: x[1])
    elem = [item[0] for item in dist]
    return elem


def get_thresholds(X, minsize, i):
    """ Get the threshold cluster for the i^{th} element of X
    :param X: Data set
    :param D: Dictionary
    :param minsize: Minimum size of a cluster
    :param i: element is X[i]
    :return:
    """
    elem = set_distances(X, {i})
    thresholds = []
    for j in range(minsize - 1, len(elem)):
        cluster = set(elem[:j])
        cluster.add(i)
        thresholds.append(cluster)
    return thresholds, i, elem


def threshold(X, e, g, s, k):
    """ Get all threshold clusters (algorithm 7, lines 1-6)
    :param X: Data matrix
    :param e: lower bound on fractional size of each cluster
    :param g: lower bound on fractional size of a set inside own cluster for which stability holds
    :param s: lower bound on fractional size of a set outside own cluster for which stability holds
    :param k: Number of clusters
    :return: Threshold clusters
    """
    print('Populating list with all threshold clusters')
    start = time.clock()
    n = len(X)
    minsize = int(e * n)
    items = Parallel(n_jobs=num_cpu)(delayed(get_thresholds)(X, minsize, i) for i in range(n))
    threshold_lists = [item[0] for item in items]
    L = [item for sublist in threshold_lists for item in sublist]
    D = dict([(item[1], item[2]) for item in items])
    end = time.clock()
    print('time = ', (end - start))
    return refine(L, X, D, e, g, s, k)


def refine(L, X, D, e, g, s, k):
    """ Throw out bad points (algorithm 7, lines 7-17)
    :param L: List of subsets
    :param X: Data matrix
    :param D: dictionary
    :param e: lower bound on fractional size of each cluster
    :param g: lower bound on fractional size of a set inside own cluster for which stability holds
    :param s: lower bound on fractional size of a set outside own cluster for which stability holds
    :param k: Number of clusters
    :return: Refined clusters
    """
    print('Getting rid of bad points')
    start = time.clock()
    n = len(X)
    T = int((e - 2*g - s*k) * n)
    t = int((e - g) * n)
    print('length of L = ' + str(len(L)))
    for i in range(len(L)):
        A = L[i]
        B = set()
        while A:
            u = A.pop()
            Cu = D[u]
            Cu = Cu[:t]
            Cu = set(Cu)
            if len(L[i].intersection(Cu)) >= T:
                B.add(u)
        L[i] = B
    end = time.clock()
    print('time = %d s' % (end - start))
    return grow(L, X, g)


def grow(L, X, g):
    """ Get back good points (algorithm 7, lines 18-21)
    :param L:
    :param X:
    :param g:
    :return:
    """
    print('Getting back good points')
    start = time.clock()
    n = len(X)
    t = int(g*n)
    for i in range(len(L)):
        A = L[i]
        elem = set_distances(X, A)
        tmp = set(elem[:t])
        A = A.union(tmp)
        L[i] = A
    end = time.clock()
    print('time = %d s' % (end - start))
    return L


def inverse_similarity(X, A, B):
    """ Compute the distance between A and B
    :param X: Data matrix
    :param A: set of points
    :param B: set of points
    :return: distance or inverse similarity
    """
    dist = 0
    for i in A:
        for j in B:
            dist += norm(X[i] - X[j])
    dist /= len(A)
    dist /= len(B)
    return dist


def non_laminar_pairs(L):
    """ Return two sets C1 and C2 in L that are not laminar, else return None
    :param L: List of subsets
    :return: Two set indices
    """
    for i in range(len(L)):
        for j in range(len(L)):
            if i == j:
                continue
            intersection = L[i].intersection(L[j])
            if len(intersection) > 0:
                if L[i].issubset(L[j]) or L[j].issubset(L[i]):
                    continue
                else:
                    return i, j
    return


def laminar(L, X, e, g, s):
    """ Make family laminar (Algorithm 9)
    :param L: List of subsets
    :param X: The data set
    :param e: lower bound on the fractional size of every cluster
    :param g: lower bound on the fractional size of every set in own cluster for which stability holds
    :param s: lower bound on the fractional size of every set in outside cluster for which stability holds
    :return: Laminar list
    """
    print('Making the list laminar')
    start = time.clock()
    n = len(X)
    S = non_laminar_pairs(L)
    while S is not None:
        i = S[0]
        j = S[1]
        intersection = L[i].intersection(L[j])
        if len(intersection) > int(s * n):
            A = intersection
            C1 = L[i].difference(A)
            C2 = L[j].difference(A)
            if inverse_similarity(X, A, C1) <= inverse_similarity(X, A, C2):
                del L[j]
            else:
                del L[i]
        else:
            # Intersection is small
            v = intersection.pop()
            elem = set_distances(X, {v})
            t = int((e - g) * n)
            elem = elem[:t]
            int1 = len(L[i].intersection(elem))
            int2 = len(L[j].intersection(elem))
            if int1 >= int2:
                del L[j]
            else:
                del L[i]
        S = non_laminar_pairs(L)
    end = time.clock()
    print('time = %d' % (end - start))
    return L


def prune(L, tcluster, k, label):
    """ Given a laminar list and a target cluster return minimum error
    :param L: Laminar list
    :param tcluster: target cluster
    :param k: number of clusters
    :param label: label of every element
    :return:
    """
    print('Pruning the tree for the best cluster')
    if len(L) == 0:
        ''' Empty list'''
        return error(label, tcluster), label
    if len(L) == 1:
        for i in L[0]:
            label[i] = k
        return error(label, tcluster), label
    if k == 1:
        # Not enough labels
        A = set()
        for item in L:
            A.union(item)
        for i in A:
            label[i] = k
        return error(label, tcluster), label
    # compute cost of including L[0] and not including L[0]
    A = L[0]
    new_list = list()
    inclusion_label = label
    # new_list contains all sets not intersecting with A
    for i in range(len(L)):
        if A & L[i]:
            # A and L[i] intersect, don't include
            continue
        else:
            # A and L[i] don't intersect
            new_list.append(L[i])
    for i in A:
        inclusion_label[i] = k
    inclusion_error = prune(new_list, tcluster, k-1, inclusion_label)
    non_inclusion_error = prune(L[1:], tcluster, k, label)
    if inclusion_error[0] < non_inclusion_error[0]:
        result = inclusion_error[0]
        label = inclusion_label
    else:
        result = non_inclusion_error[0]
    return result, label


def test(X, tcluster, k, e):
    """ Test error on a data set
    :param X: Data  matrix
    :param tcluster: Target clusters
    :param k: Number of target clusters
    :param e: Between (0, 1) represents minimum size of a cluster
    :return: None, print results
    """
    k = int(k)
    e = float(e)
    # X = np.loadtxt(name, dtype=float, delimiter=',', usecols=range(1, 14))
    y = pdist(X, metric='euclidean')
    Z = list()
    Z.append(hac.linkage(y, method='single'))
    Z.append(hac.linkage(y, method='complete'))
    Z.append(hac.linkage(y, method='average'))
    Z.append(hac.linkage(X, method='ward'))
    # tcluster = np.loadtxt(name, dtype=int, delimiter=',', usecols=(0,))
    other_clusters = [hac.fcluster(x, k, 'maxclust') for x in Z]
    errors = [error(x, tcluster) for x in other_clusters]
    s = (0.8*e)/(2*k + 1)
    g = 0.8*0.2*e
    print('k = ' + str(k))
    print('e = ' + str(e))
    print('g = ' + str(g))
    print('s = ' + str(s))
    L = threshold(X, e, g, s, k)
    L = laminar(L, X, e, g, s)
    label = [1]*len(X)
    pruned = prune(L, tcluster, k, label)
    print('Error rate = %d' % pruned[0])
    print('Error rate on other methods = ' + str(errors))
    print('Labels = ' + str(label))


if __name__ == '__main__':
    fname = 'wine.bin'
    X = np.loadtxt(fname, dtype=float, delimiter=',', usecols=range(1, 14))
    tcluster = np.loadtxt(fname, dtype=float, delimiter=',', usecols=(0,))
    print('Shape of X = ', X.shape)
    print('Shape of tcluster = ', tcluster.shape)
    k = 3
    test(X, tcluster, 3, 0.3)
