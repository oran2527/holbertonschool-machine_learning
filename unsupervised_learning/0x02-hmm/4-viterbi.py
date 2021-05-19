#!/usr/bin/env python3
""" Viterbi """
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """ Viterbi """
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

    omega = np.zeros((N, T))
    omega[:, 0] = np.multiply(Initial.T, Emission[:, Observation[0]])

    prev = np.zeros((N, T))

    for t in range(1, T):
        for j in range(N):
            # Same as Forward Probability
            omega[j, t] = np.max(omega[:, t - 1] * Transition[:, j]) * \
                      Emission[j, Observation[t]]
            prev[j, t] = np.argmax(omega[:, t - 1] * Transition[:, j])
    # Path Array
    S = np.zeros(T)
    S[T - 1] = np.argmax(omega[:, T - 1])

    for i in range(T - 2, -1, -1):
        S[i] = prev[int(S[i + 1]), i + 1]

    P = np.max(omega[:, T - 1:], axis=0)[0]
    S = [int(i) for i in S]
    return S, P
