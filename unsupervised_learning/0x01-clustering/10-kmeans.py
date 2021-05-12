#!/usr/bin/env python3
""" K-means SKLEARN """
import sklearn.cluster


def kmeans(X, k):
    """ K-means SKLEARN """
    import sklearn.cluster
    C, clss, _ = sklearn.cluster.k_means(X, k)
    return C, clss
