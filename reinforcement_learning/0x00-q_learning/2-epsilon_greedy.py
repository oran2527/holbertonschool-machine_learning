#!/usr/bin/env python3
""" Epsilon Greedy """
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """ Epsilon Greedy """
    p = np.random.uniform(0, 1)
    if p > epsilon:
        action = np.argmax(Q[state, :])
    else:
        action = np.random.randint(1)
    return action
