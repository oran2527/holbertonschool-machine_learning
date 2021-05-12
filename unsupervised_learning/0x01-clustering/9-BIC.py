#!/usr/bin/env python3
""" BIC """
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """ BIC """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None, None
    if type(kmin) is not int or kmin < 1:
        return None, None, None, None
    n, d = X.shape
    if kmax is None:
        kmax = n
    if type(kmax) is not int or kmax < 1 or kmax <= kmin:
        return None, None, None, None
    if type(iterations) is not int or iterations < 1:
        return None, None, None, None
    if type(tol) is not float or tol < 0:
        return None, None, None, None
    if type(verbose) is not bool:
        return None, None, None, None
    pi_all = []
    m_all = []
    S_all = []
    l_all = []
    b = []
    for k in range(kmin, kmax + 1):
        pi, m, S, g, like = expectation_maximization(X, k, iterations,
                                                     tol, verbose)
        pi_all.append(pi)
        m_all.append(m)
        S_all.append(S)
        l_all.append(like)
        p = k * d
        b.append(p * np.log(n) - 2 * like)
    pi_all = np.array(pi_all)
    m_all = np.array(m_all)
    S_all = np.array(S_all)
    l_all = np.array(l_all)
    b = np.array(b)
    best = np.argmin(b)
    best_result = (pi_all[best], m_all[best], S_all[best])
    return best, best_result, l_all, b
