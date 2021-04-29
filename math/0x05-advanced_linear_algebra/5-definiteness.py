#!/usr/bin/env python3
""" Definiteness """
import numpy as np


def definiteness(matrix):
    """ Definiteness """
    if type(matrix) != np.ndarray:
        raise TypeError("matrix must be a numpy.ndarray")
    if len(matrix) == 0:
        return None
    for x in matrix:
        if len(x) != len(matrix):
            return None
    if not np.allclose(matrix, matrix.T):
        return None
    w, v = np.linalg.eig(matrix)
    # count to classify
    pos = 0
    sem_pos = 0
    neg = 0
    sem_neg = 0
    for x in w:
        if x > 0:
            pos += 1
        if x >= 0:
            sem_pos += 1
        if x < 0:
            neg += 1
        if x <= 0:
            sem_neg += 1
    # All eigenvalues are > 0
    if pos == len(w):
        return "Positive definite"
    # All eigenvalues are >= 0
    if sem_pos == len(w):
        return "Positive semi-definite"
    # All eigenvalues are < 0
    if neg == len(w):
        return "Negative definite"
    # All eigenvalues are <= 0
    if sem_neg == len(w):
        return "Negative semi-definite"
    if pos and neg:
        return "Indefinite"
    else:
        return None
