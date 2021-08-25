#!/usr/bin/env python3
""" Policy gradient """
import numpy as np


def policy(matrix, weight):
    """ Policy gradient """
    z = matrix.dot(weight)
    exp = np.exp(z)
    return exp / exp.sum()


def policy_gradient(state, weight):
    """ Policy gradient """
    P = policy(state, weight)
    action = np.random.choice(len(P[0]), p=P[0])
    s = P.reshape(-1, 1)
    softmax = np.diagflat(s) - np.dot(s, s.T)
    softmax = softmax[action, :]
    dlog = softmax / P[0, action]
    gradient = state.T.dot(dlog[None, :])
    return action, gradient
