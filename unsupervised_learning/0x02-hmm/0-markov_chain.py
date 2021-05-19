#!/usr/bin/env python3
""" Markov Chain """
import numpy as np


def markov_chain(P, s, t=1):
    """ Markov Chain """
    if type(P) is not np.ndarray or len(P.shape) != 2 or len(P) != len(P[0]):
        return None
    if type(s) is not np.ndarray or len(s.shape) != 2:
        return None
    if len(s) != 1 or len(s[0]) != len(P):
        return None
    if type(t) is not int or t < 1:
        return None
    for i in range(t):
        s = np.matmul(s, P)
    return s
