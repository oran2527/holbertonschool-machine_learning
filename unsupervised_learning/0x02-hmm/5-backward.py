#!/usr/bin/env python3
""" Backward """
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """ Backward """
    if type(Observation) is not np.ndarray or len(Observation.shape) != 1:
        return None, None
    T = Observation.shape[0]
    if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
        return None, None
    N, M = Emission.shape
    if type(Transition) is not np.ndarray or len(Transition.shape) != 2:
        return None, None
    if Transition.shape != (N, N):
        return None, None
    if type(Initial) is not np.ndarray or len(Initial.shape) != 2:
        return None, None
    if Initial.shape[1] != 1:
        return None, None
    beta = np.zeros((N, T))
    beta[:, T - 1] = 1
    for t in range(T - 2, -1, -1):
        beta[:, t] = np.dot(Transition,
                            np.multiply(Emission[:, Observation[t + 1]],
                                        beta[:, t + 1]))
    P = np.dot(Initial.T, np.multiply(Emission[:, Observation[0]], beta[:, 0]))
    return P, beta
