#!/usr/bin/env python3
""" Regular """
import numpy as np


def regular(P):
    """ Regular """
    if type(P) is not np.ndarray or len(P.shape) != 2 or len(P) != len(P[0]):
        return None
    p_c = np.copy(P)
    for i in range(1000):
        p_c = np.matmul(p_c, P)
        if p_c.all() > 0:
            break
    if i == 999:
        return None
    n = P.shape[0]
    # note transpose of P to find left eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    # find index of eigenvalue = 1
    idx = np.argmin(np.abs(eigenvalues - 1))
    w = np.real(eigenvectors[:, idx]).T
    # remember to normalize eigenvector to get a probability distribution
    steady = w / np.sum(w)
    return steady.reshape((1, n))
