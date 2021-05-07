#!/usr/bin/env python3
""" PCA """
import numpy as np


def pca(X, ndim):
    """ PCA """
    X_m = X - np.mean(X, axis=0)
    u, s, vh = np.linalg.svd(X_m)
    # vh Transpose is equal to W
    W = vh[:ndim]
    T = np.matmul(X_m, W.T)
    return T
