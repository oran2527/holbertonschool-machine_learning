#!/usr/bin/env python3
""" Expectation GMM """
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """ Expectation GMM """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(pi) is not np.ndarray or len(pi.shape) != 1:
        return None, None
    if not np.isclose(pi.sum(), 1):
        return None, None
    if type(m) is not np.ndarray or len(m.shape) != 2:
        return None, None
    if type(S) is not np.ndarray or len(S.shape) != 3:
        return None, None
    n, d = X.shape
    k = pi.shape[0]
    if (k, d) != m.shape or (k, d, d) != S.shape:
        return None, None
    g = []
    for i in range(k):
        P = pdf(X, m[i], S[i]) * pi[i]
        g.append(P)
    g = np.array(g)
    likelihood = np.log(g.sum(axis=0)).sum()
    g /= g.sum(axis=0)
    return g, likelihood
