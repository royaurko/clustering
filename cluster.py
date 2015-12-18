import scipy.cluster.hierarchy as hac
import numpy as np
from scipy.spatial.distance import pdist
from scipy.linalg import norm
from munkres import Munkres
import time
import multiprocessing
import multiprocessing.pool
from multiprocessing import Process, Manager
from functools import partial
import os
import argparse
import csv
import pickle
import gzip


class NoDaemonProcess(multiprocessing.Process):
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class Pool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


def error(cluster, target_cluster):
    """ Compute error between cluster and target cluster
    :param cluster: proposed cluster
    :param target_cluster: target cluster
    :return: error
    """
    k = len(set(target_cluster))
    n = len(target_cluster)
    C = list()
    T = list()
    for i in range(1, k+1):
        tmp = {j for j in range(n) if cluster[j] == i}
        C.append(tmp)
        tmp = {j for j in range(n) if target_cluster[j] == i}
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


def compute_norm(X, i, j):
    """ Return the norm of the difference of X[i] and X[j]
    :param X: Data set
    :param i: i^{th} coordinate
    :param j: j^{th} coordinate
    :return: norm of X[i]-X[j]
    """
    return norm(X[i] - X[j])


def get_distance(X, S, i):
    """ Get average distance of i from S
    :param X: Data set
    :param S: Set S
    :param i: point i
    :return: Average distance
    """
    if i in S:
        return None
    d = 0
    for j in S:
        d += norm(X[i] - X[j])
    d /= len(S)
    return i, d


def set_distances(X, S, num_workers):
    """ Return the elements outside S sorted by distance to S
    :param X: Data matrix
    :param S: set of points
    :param num_workers: Number of workers
    :return: Elements outside S sorted by distance to S
    """
    n = len(X)
    with Pool(num_workers) as pool:
        func = partial(get_distance, X, S)
        dist = pool.map(func, range(n))
        pool.close()
        pool.join()
    dist = [item for item in dist if item is not None]
    dist = sorted(dist, key=lambda x: x[1])
    elem = [item[0] for item in dist]
    return elem


def get_thresholds(X, minsize, num_workers, i):
    """ Get the threshold cluster for the i^{th} element of X
    :param X: Data set
    :param D: Dictionary
    :param minsize: Minimum size of a cluster
    :param num_workers: Number of workers
    :param i: element is X[i]
    :return:
    """
    elem = set_distances(X, {i}, num_workers)
    thresholds = []
    for j in range(minsize - 1, len(elem)):
        cluster = set(elem[:j])
        cluster.add(i)
        thresholds.append(cluster)
    return thresholds, i, elem


def threshold(X, e, g, s, k, num_workers):
    """ Get all threshold clusters (algorithm 7, lines 1-6)
    :param X: Data matrix
    :param e: lower bound on fractional size of each cluster
    :param g: lower bound on fractional size of a set inside own cluster for which stability holds
    :param s: lower bound on fractional size of a set outside own cluster for which stability holds
    :param k: Number of clusters
    :param num_workers: Number of workers
    :return: Threshold clusters
    """
    print('Populating list with all threshold clusters')
    start = time.clock()
    n = len(X)
    minsize = int(e * n)
    with Pool(num_workers) as pool:
        func = partial(get_thresholds, X, minsize, num_workers)
        items = pool.map(func, range(n))
        pool.close()
        pool.join()
    threshold_lists = [item[0] for item in items]
    L = [item for sublist in threshold_lists for item in sublist]
    D = dict([(item[1], item[2]) for item in items])
    end = time.clock()
    print('Length of L = ', len(L))
    print('time = ', (end - start))
    return refine(L, X, D, e, g, s, k, num_workers)


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


def refine(L, X, D, e, g, s, k, num_workers):
    """ Throw out bad points (algorithm 7, lines 7-17)
    :param L: List of subsets
    :param X: Data matrix
    :param D: dictionary
    :param e: lower bound on fractional size of each cluster
    :param g: lower bound on fractional size of a set inside own cluster for which stability holds
    :param s: lower bound on fractional size of a set outside own cluster for which stability holds
    :param k: Number of clusters
    :param num_workers: Number of workers
    :return: Refined clusters
    """
    print('Getting rid of bad points')
    print('Length of L at start = ', len(L))
    start = time.clock()
    n = len(X)
    T = int((e - 2*g - s*k) * n)
    t = int((e - g) * n)
    with Pool() as pool:
        func = partial(refine_individual, D, T, t)
        L = pool.map(func, L)
        pool.close()
        pool.join()
    end = time.clock()
    print('Length of L on end = ', len(L))
    print('time = ', (end - start))
    return grow(L, X, g, num_workers)


def grow_individual(X, t, num_workers, A):
    """ Grow an individual candidate cluster
    :param X: Data set
    :param t: Threshold
    :param A: Candidate set
    :param num_workers: Number of workers
    :return:
    """
    elem = set_distances(X, A, num_workers)
    tmp = set(elem[:t])
    A = A.union(tmp)
    return A


def grow(L, X, g, num_workers):
    """ Get back good points (algorithm 7, lines 18-21)
    :param L: The list of candidate clusters
    :param X: Data set
    :param g: Parameter on stability
    :return: Refined list of candidate clusters
    """
    print('Getting back good points')
    print('Length of L at start = ', len(L))
    start = time.clock()
    n = len(X)
    t = int(g*n)
    with Pool(num_workers) as pool:
        func = partial(grow_individual, X, t, num_workers)
        L = pool.map(func, L)
        pool.close()
        pool.join()
    end = time.clock()
    print('Length of L = ', len(L))
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


def non_laminar(L, i):
    """ Return all sets in L[i+1], ..., L[n-1] that are non-laminar with respect to L[i]
    :param L: List of subsets
    :param i: index in L from which on we computer intersections
    :return: Tuple (i, list) where list is a list of indices of sets in L[i+1], ..., L[n-1] that are non-laminar
    """
    indices = list()
    for j in range(i + 1, len(L)):
        intersection = L[i].intersection(L[j])
        if intersection:
            if L[i].issubset(L[j]) or L[j].issubset(L[i]):
                continue
            else:
                t = (i, j)
                indices.append(t)
    return indices


def mark_non_laminar(L, X, e, g, s, num_workers, t):
    """
    :param L: List of candidate clusters
    :param X: Data set
    :param e: parameter on size of clusters
    :param g: parameter on similarity condition
    :param s: parameter on similarity condition
    :param num_workers: Number of workers
    :param t: Tuple (i, j) where L[i] and L[j] are non-laminar
    :return: None, mark either L[i] or L[j] None
    """
    i, j = t[0], t[1]
    if L[i] is None or L[j] is None:
        return
    n = len(X)
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
        elem = set_distances(X, {v}, num_workers)
        t = int((e - g) * n)
        elem = elem[:t]
        int1 = len(L[i].intersection(elem))
        int2 = len(L[j].intersection(elem))
        if int1 >= int2:
            L[j] = None
        else:
            L[i] = None


def iterate_laminar(L, X, e, g, s, num_workers, intersections):
    """
    :param L: List of candidate clusters
    :param X: data set
    :param e: parameter
    :param g: parameter
    :param s: parameter
    :param num_workers: number of workers
    :param intersections: List of intersections
    """
    for item in intersections:
        mark_non_laminar(L, X, e, g, s, num_workers, item)


def laminar(L, X, e, g, s, num_workers):
    """ Make family laminar (Algorithm 9)
    :param L: List of subsets
    :param X: The data set
    :param e: lower bound on the fractional size of every cluster
    :param g: lower bound on the fractional size of every set in own cluster for which stability holds
    :param s: lower bound on the fractional size of every set in outside cluster for which stability holds
    :return: Laminar list
    """
    print('Making the list laminar (parallel)')
    start = time.clock()
    n = len(X)
    print('Computing pairs of non-laminar sets')
    with Pool(num_workers) as pool:
        func = partial(non_laminar, L)
        intersections = pool.map(func, range(len(L)-1))
        pool.close()
        pool.join()
    intersections = [item for sub_list in intersections for item in sub_list]
    end = time.clock()
    print('Length of intersections = ', len(intersections))
    print('time = ', end - start)
    print('Removing non-laminar pairs')
    start = time.clock()
    manager = Manager()
    shared_L = manager.list(L)
    process = Process(target=iterate_laminar, args=(shared_L, X, e, g, s, num_workers, intersections))
    process.start()
    process.join()
    L = [item for item in shared_L if item is not None]
    end = time.clock()
    print('Length of list after removing non-laminar pairs = ', len(L))
    print('time = ', (end - start))
    return L


def prune(L, target_cluster, k, label):
    """ Given a laminar list and a target cluster return minimum error
    :param L: Laminar list
    :param target_cluster: target cluster
    :param k: number of clusters
    :param label: label of every element
    :return:
    """
    if not L:
        #  Empty list
        return error(label, target_cluster), label
    if len(L) == 1:
        for i in L[0]:
            label[i] = k
        return error(label, target_cluster), label
    if k == 1:
        # Not enough labels
        A = set()
        for item in L:
            A.union(item)
        for i in A:
            label[i] = k
        return error(label, target_cluster), label
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
    inclusion_error = prune(new_list, target_cluster, k-1, inclusion_label)
    non_inclusion_error = prune(L[1:], target_cluster, k, label)
    if inclusion_error[0] < non_inclusion_error[0]:
        result = inclusion_error[0]
        label = inclusion_label
    else:
        result = non_inclusion_error[0]
    return result, label


def test(X, target_cluster, k, e, num_workers):
    """ Test error on a data set
    :param X: Data  matrix
    :param target_cluster: Target clusters
    :param k: Number of target clusters
    :param e: Between (0, 1) represents minimum size of a cluster
    :param num_workers: Number of workers
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
    errors = [error(x, target_cluster) for x in other_clusters]
    error_dict = {'single linkage': errors[0], 'complete linkage': errors[1], 'average linkage': errors[2], 'ward': errors[3]}
    s = (0.8*e)/(2*k + 1)
    g = 0.8*0.2*e
    print('k = ', k)
    print('e = ', e)
    print('g = ', g)
    print('s = ', s)
    L = threshold(X, e, g, s, k, num_workers)
    L = laminar(L, X, e, g, s, num_workers)
    with open('laminar.pkl', 'wb') as f:
        pickle.dump(laminar_L, f)
    label = [1]*len(X)
    print('Pruning the tree for the best cluster')
    pruned = prune(L, target_cluster, k, label)
    error_dict['threshold'] = pruned[0]
    return error_dict


def main(file_name, data_label, num_workers):
    """
    :param file_name: Name of file containing data
    :param data_label: column of data where label is
    :param num_workers: number of workers
    """
    result = 'results.pkl'
    reader = csv.reader(open(file_name), delimiter=',')
    X = list()
    target_cluster = list()
    for row in reader:
        if row:
            label = row[data_label]
            row.pop(data_label)
            X.append(row)
            target_cluster.append(label)
    X = np.array(X, dtype=float)
    target_cluster = np.array(target_cluster, dtype=int)
    k = len(set(target_cluster))
    error_dict = test(X, target_cluster, k, 1/(3*k), num_workers)
    print(error_dict)
    error_dict = str(error_dict) + '\n'
    d = dict()
    if os.path.exists(result):
        with open(result, 'rb') as f:
            try:
                d = pickle.load(f)
            except EOFError:
                pass
    with open(result, 'wb') as f:
        if file_name in d:
            d[file_name].append(error_dict)
        else:
            d[file_name] = [error_dict]
        pickle.dump(d, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', help='Data file')
    parser.add_argument('-l', '--label', type=int, default=0, help='Column of labels')
    parser.add_argument('-n', '--num_workers', type=int, default=1, help='Number of workers')
    args = parser.parse_args()
    main(args.data, args.label, args.num_workers)
