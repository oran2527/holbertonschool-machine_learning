#!/usr/bin/env python3
""" RNN """
import numpy as np


def rnn(rnn_cell, X, h_0):
    """ RNN """
    T, m, i = X.shape
    _, h = h_0.shape
    H = np.zeros((T + 1, m, h))
    Y = np.zeros((T, m, rnn_cell.Wy.shape[1]))
    H[0] = h_0
    for t in range(T):
        H[t + 1], Y[t] = rnn_cell.forward(H[t], X[t])
    return H, Y
