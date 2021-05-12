#!/usr/bin/env python3
""" Initialize centroids """
import numpy as np


def initialize(X, k):
    """ Initialize centroids """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(k) is not int or k < 1:
        return None
    n, d = X.shape
    min = np.amin(X, axis=0)
    max = np.amax(X, axis=0)
    centroids = np.random.uniform(low=min, high=max, size=(k, d))
    return centroids
