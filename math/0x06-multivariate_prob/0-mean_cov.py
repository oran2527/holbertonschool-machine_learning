#!/usr/bin/env python3
""" Mean and Covariance """
import numpy as np


def mean_cov(X):
    """ Mean and Covariance """
    if type(X) is not np.ndarray:
        raise TypeError("X must be a 2D numpy.ndarray")
    if len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")
    mean = X.mean(axis=0)
    covs = np.dot(X.T, (X - mean)) / (n - 1)
    return mean.reshape(1, d), covs
