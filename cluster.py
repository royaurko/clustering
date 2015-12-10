import scipy.cluster.hierarchy as hac
import numpy as np
from scipy.spatial.distance import pdist
from scipy.linalg import norm
from munkres import Munkres
import time
import multiprocessing
import multiprocessing.pool
from functools import partial
from functools import reduce
from contextlib import closing
import os
import argparse
import csv
num_cpu = multiprocessing.cpu_count()


class NoDaemonProcess(multiprocessing.Process):
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class Pool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


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


def get_distance(X, S, i):
    if i in S:
        return None
    S = list(S)
    norms = map(lambda j: norm(X[i] - X[j]), S)
    d = reduce(lambda x, y: x + y, norms)
    d /= len(S)
    return i, d


def set_distances(X, S):
    """ Return the elements outside S sorted by distance to S
    :param X: Data matrix
    :param S: set of points
    :return: Elements outside S sorted by distance to S
    """
    n = len(X)
    with closing(Pool(processes=num_cpu)) as pool:
        func = partial(get_distance, X, S)
        dist = pool.map(func, range(n))
        pool.close()
        pool.join()
    dist = [item for item in dist if item is not None]
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
    with closing(Pool(processes=num_cpu)) as pool:
        func = partial(get_thresholds, X, minsize)
        items = pool.map(func, range(n))
        pool.close()
        pool.join()
    threshold_lists = [item[0] for item in items]
    L = [item for sublist in threshold_lists for item in sublist]
    D = dict([(item[1], item[2]) for item in items])
    end = time.clock()
    print('time = ', (end - start))
    return refine(L, X, D, e, g, s, k)


def refine_individual(D, T, t, S):
    """ Refine a individual candidate cluster
    :param D: Dictionary
    :param T: Threshold
    :param t: threshold
    :param S: Candidate set
    :return:
    """
    A = S
    B = set()
    while A:
        u = A.pop()
        Cu = D[u]
        Cu = Cu[:t]
        Cu = set(Cu)
        if len(S.intersection(Cu)) >= T:
            B.add(u)
    return B


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
    with closing(Pool(processes=num_cpu)) as pool:
        func = partial(refine_individual, D, T, t)
        L = pool.map(func, L)
        pool.close()
        pool.join()
    end = time.clock()
    print('time = ', (end - start))
    return grow(L, X, g)


def grow_individual(X, t, A):
    """ Grow an individual candidate cluster
    :param X: Data set
    :param t: Threshold
    :param A: Candidate set
    :return:
    """
    elem = set_distances(X, A)
    tmp = set(elem[:t])
    A = A.union(tmp)
    return A


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
    with closing(Pool(processes=num_cpu)) as pool:
        func = partial(grow_individual, X, t)
        L = pool.map(func, L)
    end = time.clock()
    print('time = ', (end - start))
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


def non_laminar_pairs(L, i):
    """ Return all sets in L[i+1], ..., L[n-1] that are non-laminar with respect to L[i]
    :param L: List of subsets
    :return: Tuple (i, list) where list is a list of indices of sets in L[i+1], ..., L[n-1] that are non-laminar
    """
    indices = list()
    for j in range(i + 1, len(L)):
        intersection = L[i].intersection(L[j])
        if len(intersection) > 0:
            if L[i].issubset(L[j]) or L[j].issubset(L[i]):
                continue
            else:
                indices.append(j)
    tuples = [(i, index) for index in indices]
    return tuples


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
    with closing(Pool(processes=num_cpu)) as pool:
        func = partial(non_laminar_pairs, L)
        intersections = pool.map(func, range(n-1))
    intersections = set([item for sublist in intersections for item in sublist])
    while intersections:
        i, j = intersections.pop()
        if L[i] is None or L[j] is None:
            continue
        intersection = L[i].intersection(L[j])
        if len(intersection) > int(s * n):
            A = intersection
            C1 = L[i].difference(A)
            C2 = L[j].difference(A)
            if inverse_similarity(X, A, C1) <= inverse_similarity(X, A, C2):
                L[j] = None
            else:
                L[i] = None
        else:
            # Intersection is small
            v = intersection.pop()
            elem = set_distances(X, {v})
            t = int((e - g) * n)
            elem = elem[:t]
            int1 = len(L[i].intersection(elem))
            int2 = len(L[j].intersection(elem))
            if int1 >= int2:
                L[j] = None
            else:
                L[i] = None
    L = [item for item in L if item is not None]
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
    y = pdist(X, metric='euclidean')
    Z = list()
    Z.append(hac.linkage(y, method='single'))
    Z.append(hac.linkage(y, method='complete'))
    Z.append(hac.linkage(y, method='average'))
    Z.append(hac.linkage(X, method='ward'))
    other_clusters = [hac.fcluster(x, k, 'maxclust') for x in Z]
    errors = [error(x, tcluster) for x in other_clusters]
    error_dict = {'single linkage': errors[0], 'complete linkage': errors[1], 'average linkage': errors[2], 'ward': errors[3]}
    s = (0.8*e)/(2*k + 1)
    g = 0.8*0.2*e
    print('k = ', k)
    print('e = ', e)
    print('g = ', g)
    print('s = ', s)
    L = threshold(X, e, g, s, k)
    L = laminar(L, X, e, g, s)
    label = [1]*len(X)
    pruned = prune(L, tcluster, k, label)
    error_dict['threshold'] = pruned[0]
    return error_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', help='Data file')
    parser.add_argument('-l', '--label', type=int, default=0, help='Column of labels')
    args = parser.parse_args()
    fname = args.data
    result_dir = 'results'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    out_file_name = result_dir + '/' + fname + '_' + time_str
    f = open(out_file_name, 'wb')
    reader = csv.reader(open(fname), delimiter=',')
    X = list()
    tcluster = list()
    for row in reader:
        if row:
            label = row[args.label]
            row.pop(args.label)
            X.append(row)
            tcluster.append(label)
    X = np.array(X, dtype=float)
    print('Shape of X = ', X.shape)
    print('Length of tcluster = ', len(tcluster))
    k = len(set(tcluster))
    error_dict = test(X, tcluster, k, 1/(3*k))
    error_dict = str(error_dict) + '\n'
    f.write(bytes(error_dict, 'utf-8'))
    f.close()
