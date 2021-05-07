#!/usr/bin/env python3
""" PCA """
import numpy as np


def pca(X, var=0.95):
    """ PCA """
    u, s, vh = np.linalg.svd(X)
    # accumulative sum of eigenvalues
    cum = np.cumsum(s) / sum(s)
    # Get indices less than variance
    idx = np.where(cum <= var, 1, 0)
    n = sum(idx) + 1
    # vh Transpose is equal to W
    return vh.T[:, :n]
