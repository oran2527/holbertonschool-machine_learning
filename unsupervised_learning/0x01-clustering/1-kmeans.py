#!/usr/bin/env python3
""" K-means """
import numpy as np


def closest_centroid(X, centroids):
    """ K-means """
    distances = np.linalg.norm(X - centroids[:, np.newaxis], axis=2)
    return np.argmin(distances, axis=0)


def kmeans(X, k, iterations=1000):
    """ K-means """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(k) is not int or k < 1:
        return None, None
    if type(iterations) is not int or iterations < 1:
        return None, None
    n, d = X.shape
    min = np.amin(X, axis=0)
    max = np.amax(X, axis=0)
    centroids = np.random.uniform(low=min, high=max, size=(k, d))
    clss = closest_centroid(X, centroids)
    for i in range(iterations):
        copy = np.copy(centroids)
        for c in range(k):
            idx = np.where(clss == c)
            if len(idx[0]) == 0:
                centroids[c] = np.random.uniform(min, max, (1, d))
            else:
                mean = X[idx].mean(axis=0)
                centroids[c] = mean
        clss = closest_centroid(X, centroids)
        if np.array_equal(copy, centroids):
            break
    return centroids, clss
