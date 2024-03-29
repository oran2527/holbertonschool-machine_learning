#!/usr/bin/env python3
""" EM Algorithm """
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """ EM Algorithm """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None, None, None
    if type(k) is not int or k < 1:
        return None, None, None, None, None
    if type(iterations) is not int or iterations < 1:
        return None, None, None, None, None
    if type(tol) is not float or tol < 0:
        return None, None, None, None, None
    if type(verbose) is not bool:
        return None, None, None, None, None
    pi, m, S = initialize(X, k)
    l_prev = 0
    for i in range(iterations):
        g, like = expectation(X, pi, m, S)
        if verbose:
            if i % 10 == 0 or abs(like - l_prev) <= tol or i + 1 == iterations:
                print("Log Likelihood after {} "
                      "iterations: {}".format(i, round(like, 5)))
        pi, m, S = maximization(X, g)
        if abs(like - l_prev) <= tol:
            break
        l_prev = like
    return pi, m, S, g, like
