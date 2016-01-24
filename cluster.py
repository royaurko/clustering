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
    C = []
    T = []
    for i in range(1, k+1):
        tmp = {j for j in range(n) if cluster[j] == i}
        C.append(tmp)
        tmp = {j for j in range(n) if target_cluster[j] == i}
        T.append(tmp)
    M = []
    for i in range(k):
        M.append([0]*k)
    testM = []
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


def get_avg_distance(X, S, i):
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


def get_max_distance(X, S, i):
    """ Get max distance of i from S
    :param X: Data set
    :param S: Set S
    :param i: point i
    :return: Max distance
    """
    if i in S:
        return None
    d = max([norm(X[i] - X[j]) for j in S])
    return i, d


def get_min_distance(X, S, i):
    """ Get min distance of i from S
    :param X: Data set
    :param S: Set S
    :param i: point i
    :return: Min distance
    """
    if i in S:
        return None
    d = min([norm(X[i] - X[j]) for j in S])
    return i, d


def set_distances(X, S, num_workers, metric):
    """ Return the elements outside S sorted by distance to S
    :param X: Data matrix
    :param S: set of points
    :param num_workers: Number of workers
    :param metric: metric is in the set {avg, min, max}
    :return: Elements outside S sorted by distance to S
    """
    n = len(X)
    if metric == 'avg':
        get_distance = get_avg_distance
    elif metric == 'max':
        get_distance = get_max_distance
    else:
        get_distance = get_min_distance
    with Pool(num_workers) as pool:
        func = partial(get_distance, X, S)
        dist = pool.map(func, range(n))
        pool.close()
        pool.join()
    dist = [item for item in dist if item is not None]
    dist = sorted(dist, key=lambda x: x[1])
    elem = [item[0] for item in dist]
    return elem


def get_thresholds(X, minsize, num_workers, metric, i):
    """ Get the threshold cluster for the i^{th} element of X
    :param X: Data set
    :param D: Dictionary
    :param minsize: Minimum size of a cluster
    :param num_workers: Number of workers
    :param metric: metric is in the set {avg, min, max}
    :param i: element is X[i]
    :return:
    """
    elem = set_distances(X, {i}, num_workers, metric)
    thresholds = []
    for j in range(minsize - 1, len(elem)):
        cluster = set(elem[:j])
        cluster.add(i)
        thresholds.append(cluster)
    return thresholds, i, elem


def threshold(X, e, a, b, k, num_workers, metric):
    """ Get all threshold clusters (algorithm 7, lines 1-6)
    :param X: Data matrix
    :param e: lower bound on fractional size of each cluster
    :param a: lower bound on fractional size of a set inside own cluster for which stability holds
    :param b: lower bound on fractional size of a set outside own cluster for which stability holds
    :param k: Number of clusters
    :param num_workers: Number of workers
    :param metric: metric is in the set {avg, min, max}
    :return: Threshold clusters
    """
    print('Populating list with all threshold clusters with metric:', metric)
    start = time.time()
    n = len(X)
    minsize = int(e * n)
    with Pool(num_workers) as pool:
        func = partial(get_thresholds, X, minsize, num_workers, metric)
        items = pool.map(func, range(n))
        pool.close()
        pool.join()
    threshold_lists = [item[0] for item in items]
    L = [item for sublist in threshold_lists for item in sublist]
    D = dict([(item[1], item[2]) for item in items])
    end = time.time()
    print('Length of L = ', len(L))
    print('time = {0:.2f}s'.format(end - start))
    return refine(L, X, D, e, a, b, k, num_workers, metric)


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


def refine(L, X, D, e, a, b, k, num_workers, metric):
    """ Throw out bad points (algorithm 7, lines 7-17)
    :param L: List of subsets
    :param X: Data matrix
    :param D: dictionary
    :param e: lower bound on fractional size of each cluster
    :param a: lower bound on fractional size of a set inside own cluster for which stability holds
    :param b: lower bound on fractional size of a set outside own cluster for which stability holds
    :param k: Number of clusters
    :param num_workers: Number of workers
    :param metric: metric is in {avg, max, min}
    :return: Refined clusters
    """
    print('Getting rid of bad points')
    print('Length of L at start = ', len(L))
    start = time.time()
    n = len(X)
    T = int((e - 2*a - b*k) * n)
    t = int((e - a) * n)
    with Pool() as pool:
        func = partial(refine_individual, D, T, t)
        L = pool.map(func, L)
        pool.close()
        pool.join()
    end = time.time()
    print('Length of L on end = ', len(L))
    print('time = {0:.2f}s'.format(end - start))
    return grow(L, X, a, num_workers, metric)


def grow_individual(X, t, num_workers, metric, A):
    """ Grow an individual candidate cluster
    :param X: Data set
    :param t: Threshold
    :param metric: metric is in {avg, max, min}
    :param A: Candidate set
    :param num_workers: Number of workers
    :return:
    """
    elem = set_distances(X, A, num_workers, metric)
    tmp = set(elem[:t])
    A = A.union(tmp)
    return A


def grow(L, X, a, num_workers, metric):
    """ Get back good points (algorithm 7, lines 18-21)
    :param L: The list of candidate clusters
    :param X: Data set
    :param a: Parameter on stability
    :param num_workers: Number of workers
    :param metric: metric is in {avg, max, min}
    :return: Refined list of candidate clusters
    """
    print('Getting back good points')
    print('Length of L at start = ', len(L))
    start = time.time()
    n = len(X)
    t = int(a*n)
    with Pool(num_workers) as pool:
        func = partial(grow_individual, X, t, num_workers, metric)
        L = pool.map(func, L)
        pool.close()
        pool.join()
    end = time.time()
    print('Length of L = ', len(L))
    print('time = {0:.2f}s'.format(end - start))
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
    indices = []
    for j in range(i + 1, len(L)):
        intersection = L[i].intersection(L[j])
        if intersection:
            if L[i].issubset(L[j]) or L[j].issubset(L[i]):
                continue
            else:
                t = (i, j)
                indices.append(t)
    return indices


def mark_non_laminar(L, X, e, a, b, num_workers, metric, t):
    """
    :param L: List of candidate clusters
    :param X: Data set
    :param e: parameter on size of clusters
    :param a: parameter on similarity condition
    :param b: parameter on similarity condition
    :param num_workers: Number of workers
    :param metric: metric is in {avg, max, min}
    :param t: Tuple (i, j) where L[i] and L[j] are non-laminar
    :return: None, mark either L[i] or L[j] None
    """
    i, j = t[0], t[1]
    n = len(X)
    try:
        intersection = L[i].intersection(L[j])
    except:
        return
    if len(intersection) > int(b * n):
        A = intersection
        try:
            C1 = L[i].difference(A)
            C2 = L[j].difference(A)
        except:
            return
        if inverse_similarity(X, A, C1) <= inverse_similarity(X, A, C2):
            L[j] = None
        else:
            L[i] = None
    else:
        # Intersection is small
        v = intersection.pop()
        elem = set_distances(X, {v}, num_workers, metric)
        t = int((e - a) * n)
        elem = elem[:t]
        try:
            int1 = len(L[i].intersection(elem))
            int2 = len(L[j].intersection(elem))
        except:
            return
        if int1 >= int2:
            L[j] = None
        else:
            L[i] = None


def iterate_laminar(L, X, e, a, b, num_workers, metric, intersections):
    """
    :param L: List of candidate clusters
    :param X: data set
    :param e: parameter
    :param a: parameter
    :param b: parameter
    :param num_workers: number of workers
    :param metric: metric is in {avg, max, min}
    :param intersections: List of intersections
    """
    for item in intersections:
        mark_non_laminar(L, X, e, a, b, num_workers, metric, item)


def laminar(L, X, e, a, b, num_workers, metric):
    """ Make family laminar (Algorithm 9)
    :param L: List of subsets
    :param X: The data set
    :param e: lower bound on the fractional size of every cluster
    :param a: lower bound on the fractional size of every set in own cluster for which stability holds
    :param b: lower bound on the fractional size of every set in outside cluster for which stability holds
    :param num_workers: number of workers
    :param metric: metric is in {avg, max, min}
    :return: Laminar list
    """
    print('Making the list laminar (parallel)')
    start = time.time()
    n = len(X)
    print('Computing pairs of non-laminar sets')
    with Pool(num_workers) as pool:
        func = partial(non_laminar, L)
        intersections = pool.map(func, range(len(L)-1))
        pool.close()
        pool.join()
    intersections = [item for sub_list in intersections for item in sub_list]
    end = time.time()
    fname = 'intersections_' + metric + '.pkl.gz'
    # with gzip.open(fname, 'wb') as f:
    #    pickle.dump(intersections, f)
    print('Length of intersections = ', len(intersections))
    print('time = {0:0.2f}s'.format(end - start))
    print('Removing non-laminar pairs')
    start = time.time()
    manager = Manager()
    shared_L = manager.list(L)
    n = len(intersections)
    j = 0
    batch = int(n/num_workers)
    rem = n % num_workers
    jobs = []
    for i in range(num_workers):
        process = Process(target=iterate_laminar, args=(shared_L, X, e, a, b, num_workers, metric, intersections[j: j + batch]))
        process.start()
        jobs.append(process)
        j += batch
    if rem:
        process = Process(target=iterate_laminar, args=(shared_L, X, e, a, b, num_workers, metric, intersections[j: j + rem]))
        process.start()
        jobs.append(process)
    for p in jobs:
        p.join()
    L = [item for item in shared_L if item is not None]
    end = time.time()
    print('Length of list after removing non-laminar pairs = ', len(L))
    print('time = {0:.2f}s'.format(end - start))
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
    new_list = []
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


def test(X, target_cluster, params, metric, num_workers):
    """ Test error on a data set
    :param X: Data  matrix
    :param target_cluster: Target clusters
    :param params: contains the parameters of the algorithm
    :param metric: Metric is in {avg, max, min}
    :param num_workers: Number of workers
    :return: None, print results
    """
    k = params['k']
    e = params['e']
    a = params['a']
    b = params['b']
    print('k = ', k)
    print('e = ', e)
    print('a = ', a)
    print('b = ', b)
    y = pdist(X, metric='euclidean')
    Z = []
    Z.append(hac.linkage(y, method='single'))
    Z.append(hac.linkage(y, method='complete'))
    Z.append(hac.linkage(y, method='average'))
    Z.append(hac.linkage(X, method='ward'))
    other_clusters = [hac.fcluster(x, k, 'maxclust') for x in Z]
    errors = [error(x, target_cluster) for x in other_clusters]
    error_dict = {'single linkage': errors[0], 'complete linkage': errors[1], 'average linkage': errors[2], 'ward': errors[3]}
    L = threshold(X, e, a, b, k, num_workers, metric)
    prelaminar_name = 'prelaminar_' + metric
    with open(prelaminar_name, 'wb') as f:
        pickle.dump(L, f)
    L = laminar(L, X, e, a, b, num_workers, metric)
    laminar_name = 'laminar_' + metric
    with open(laminar_name, 'wb') as f:
        pickle.dump(L, f)
    label = [1]*len(X)
    print('Pruning the tree for the best cluster')
    pruned = prune(L, target_cluster, k, label)
    threshold_key = 'threshold_' + metric
    error_dict[threshold_key] = pruned[0]
    print('Error on metric: {} is {}'.format(metric, pruned[0]))
    return error_dict


def main(data, target, metric, out_file, num_workers):
    """
    :param data: Data
    :param target: target (numerical)
    :param metric: Metric is in {avg, max, min}
    :param out_file: Name of pickle file to store result
    :param num_workers: number of workers
    """
    if metric not in {'avg', 'max', 'min'}:
        return
    k = len(set(target))
    e = 1/(2*k)
    # Create the params dictionary to pass to test()
    params = {'k': k, 'e': e, 'b': (0.8*e)/(2*k + 2), 'a': 0.8*0.1*e}
    error_dict = test(data, target, params, metric, num_workers)
    error_dict['params'] = params
    print('Errors = ', error_dict)
    with open(out_file, 'wb') as f:
        pickle.dump(error_dict, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', help='Data file')
    parser.add_argument('-l', '--label', type=int, default=0, help='Column of labels')
    parser.add_argument('-m', '--metric', default='avg', help='Can be one of {avg, max, min}')
    parser.add_argument('-o', '--out_file', default='result.pkl', help='Pickle file to store the result')
    parser.add_argument('-n', '--num_workers', type=int, default=1, help='Number of workers')
    args = parser.parse_args()
    reader = csv.reader(open(args.data), delimiter=',')
    data = []
    target = []
    for row in reader:
        if row:
            label = row[args.label]
            row.pop(args.label)
            data.append(row)
            target.append(label)
    data = np.array(data, dtype=float)
    labels = set(target)
    label_to_idx = {v: i for i, v in enumerate(labels)}
    target = np.array([label_to_idx[i] for i in target], dtype=int)
    main(data, target, args.metric, args.out_file, args.num_workers)
