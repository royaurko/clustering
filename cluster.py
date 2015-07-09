#!/usr/bin/python
import numpy as np
import scipy.cluster.hierarchy as hac
from scipy.spatial.distance import pdist
from scipy.linalg import norm
from munkres import Munkres
import time


def error(cluster, tcluster):
    # Compute error between cluster and target cluster
    k = len(set(tcluster))
    total = len(cluster)
    n = len(tcluster)
    C = list()
    T = list()
    for i in range(1, k+1):
        tmp = set([j for j in range(n) if cluster[j] == i])
        C.append(tmp)
        tmp = set([j for j in range(n) if tcluster[j] == i])
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
    assert total == total2
    return float(total)/float(n)


def setdistances(X, S):
    '''Return the elements outside S sorted by distance to S'''
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


def threshold(X, e, g, s, k):
    ''' Get all threshold clusters (algorithm 7, lines 1-6)'''
    print 'Populating list with all threshold clusters'
    start = time.clock()
    D = dict()
    L = list()
    n = len(X)
    minsize = int(e * n)
    for i in range(n):
        elem = setdistances(X, set([i]))
        D[i] = elem
        for j in range(minsize - 1, len(elem)):
            cluster = set(elem[:j])
            cluster.add(i)
            L.append(cluster)
    end = time.clock()
    print 'time = %d s' % (end - start)
    return refine(L, X, D, e, g, s, k)


def refine(L, X, D, e, g, s, k):
    ''' Throw out bad points (algorithm 7, lines 7-17)'''
    print 'Getting rid of bad points'
    start = time.clock()
    n = len(X)
    T = int((e - 2*g - s*k) * n)
    t = int((e - g) * n)
    print 'length of L = ' + str(len(L))
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
    print 'time = %d s' % (end - start)
    return grow(L, X, g)


def grow(L, X, g):
    ''' Get back good points (algorithm 7, lines 18-21)'''
    print 'Getting back good points'
    start = time.clock()
    n = len(X)
    t = int(g*n)
    for i in range(len(L)):
        A = L[i]
        elem = setdistances(X, A)
        tmp = set(elem[:t])
        A = A.union(tmp)
        L[i] = A
    end = time.clock()
    print 'time = %d s' % (end - start)
    return L


def invsimilarity(X, A, B):
    '''Compute the distance between A and B'''
    dist = 0
    for i in A:
        for j in B:
            dist += norm(X[i] - X[j])
    dist /= len(A)
    dist /= len(B)
    return dist


def nonlaminarpairs(L):
    '''Return two sets C1 and C2 in L that are not laminar, else return None'''
    for i in range(len(L)):
        for j in range(len(L)):
            if i == j:
                continue
            intersection = L[i].intersection(L[j])
            if len(intersection) > 0:
                if L[i].issubset(L[j]) or L[j].issubset(L[i]):
                    continue
                else:
                    return (i, j)
    return


def laminar(L, X, e, g, s):
    ''' Make family laminar (Algorithm 9)'''
    print 'Making the list laminar'
    start = time.clock()
    n = len(X)
    S = nonlaminarpairs(L)
    while S is not None:
        i = S[0]
        j = S[1]
        intersection = L[i].intersection(L[j])
        if len(intersection) > int(s * n):
            A = intersection
            C1 = L[i].difference(A)
            C2 = L[j].difference(A)
            if invsimilarity(X, A, C1) <= invsimilarity(X, A, C2):
                del L[j]
            else:
                del L[i]
        else:
            # Intersection is small
            v = intersection.pop()
            elem = setdistances(X, set([v]))
            t = int((e - g) * n)
            elem = elem[:t]
            int1 = len(L[i].intersection(elem))
            int2 = len(L[j].intersection(elem))
            if int1 >= int2:
                del L[j]
            else:
                del L[i]
        S = nonlaminarpairs(L)
    end = time.clock()
    print 'time = %d' % (end - start)
    return L


def prune(L, tcluster, k, label):
    '''Given a laminar list and a target cluster return minimum error'''
    print 'Pruning the tree for the best cluster'
    if len(L) == 0:
        ''' Empty list'''
        return (error(label, tcluster), label)
    if len(L) == 1:
        for i in L[0]:
            label[i] = k
        return (error(label, tcluster), label)
    if k == 1:
        # Not enough labels
        A = set()
        for item in L:
            A.union(item)
        for i in A:
            label[i] = k
        return (error(label, tcluster), label)
    # compute cost of including L[0] and not including L[0]
    A = L[0]
    newL = list()
    inclusionlabel = label
    # newL contains all sets not intersecting with A
    for i in range(len(L)):
        if A & L[i]:
            '''A and L[i] intersect, don't include'''
            continue
        else:
            '''A and L[i] don't intersect'''
            newL.append(L[i])
    for i in A:
        inclusionlabel[i] = k
    inclusionerror = prune(newL, tcluster, k-1, inclusionlabel)
    noninclusionerror = prune(L[1:], tcluster, k, label)
    if inclusionerror[0] < noninclusionerror[0]:
        result = inclusionerror[0]
        label = inclusionlabel
    else:
        result = noninclusionerror[0]
    return (result, label)


def test(fname, k, e):
    '''arg1 = filename, arg2 = k, arg3 = e, tests error on this data'''
    k = int(k)
    e = float(e)
    X = np.loadtxt(fname, dtype=float, delimiter=',', usecols=range(1, 14))
    y = pdist(X, metric='euclidean')
    Z = list()
    Z.append(hac.linkage(y, method='single'))
    Z.append(hac.linkage(y, method='complete'))
    Z.append(hac.linkage(y, method='average'))
    Z.append(hac.linkage(X, method='ward'))
    tcluster = np.loadtxt(fname, dtype=int, delimiter=',', usecols=(0,))
    otherclusters = map(lambda x: hac.fcluster(x, k, 'maxclust'), Z)
    errors = map(lambda x: error(x, tcluster), otherclusters)
    s = (0.8*e)/(2*k + 1)
    g = 0.8*0.2*e
    print 'k = ' + str(k)
    print 'e = ' + str(e)
    print 'g = ' + str(g)
    print 's = ' + str(s)
    L = threshold(X, e, g, s, k)
    L = laminar(L, X, e, g, s)
    label = [1]*len(X)
    pruned = prune(L, tcluster, k, label)
    print 'Error rate = %d' % pruned[0]
    print 'Error rate on other methods = ' + str(errors)
    print 'Labels = ' + str(label)
