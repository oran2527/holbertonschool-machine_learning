#!/usr/bin/env python3
""" Agglomerative """
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """ Agglomerative """
    h = scipy.cluster.hierarchy
    Z = h.linkage(X, 'ward')
    f = h.fcluster(Z, dist, 'distance')
    h.dendrogram(Z)
    plt.show()
    return f
