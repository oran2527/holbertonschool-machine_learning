#!/usr/bin/env python3
""" Correlation """
import numpy as np


def correlation(C):
    """ Correlation """
    if type(C) is not np.ndarray:
        raise TypeError("C must be a numpy.ndarray")
    if len(C.shape) != 2:
        raise ValueError("C must be a 2D square matrix")
    if C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")
    d = np.sqrt(np.diag(C))
    outer_v = np.outer(d, d)
    cor = C / outer_v
    cor[C == 0] = 0
    return cor
