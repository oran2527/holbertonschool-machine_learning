#!/usr/bin/env python3
""" Intersection with prior """
import numpy as np


def likelihood(x, n, P):
    """ Intersection with prior """
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        raise ValueError("x must be an integer that is greater than or"
                         " equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    for i in P:
        if i < 0 or i > 1:
            raise ValueError("All values in P must be in the range [0, 1]")
    fact = np.math.factorial
    a = fact(n) / (fact(x) * fact(n - x))
    b = (P ** x) * ((1 - P) ** (n - x))
    return a * b


def intersection(x, n, P, Pr):
    """ Intersection with prior """
    if type(n) is not int or n < 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        raise ValueError("x must be an integer that is greater than or"
                         " equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if type(Pr) is not np.ndarray or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    for i, j in zip(P, Pr):
        if i < 0 or i > 1:
            raise ValueError("All values in P must be in the range [0, 1]")
        if j < 0 or j > 1:
            raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")
    L = likelihood(x, n, P)
    return L * Pr


def marginal(x, n, P, Pr):
    """ Intersection with prior """
    if type(n) is not int or n < 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        raise ValueError("x must be an integer that is greater than or"
                         " equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if type(Pr) is not np.ndarray or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    for i, j in zip(P, Pr):
        if i < 0 or i > 1:
            raise ValueError("All values in P must be in the range [0, 1]")
        if j < 0 or j > 1:
            raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")
    a = intersection(x, n, P, Pr)
    return np.sum(a)


def posterior(x, n, P, Pr):
    """ Intersection with prior """ 
    int = intersection(x, n, P, Pr)
    marg = marginal(x, n, P, Pr)
    post = int / marg
    return post
