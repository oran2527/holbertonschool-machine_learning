#!/usr/bin/env python3
""" Bidirectional RNN """
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """ Bidirectional RNN """
    T, m, i = X.shape
    _, h = h_0.shape
    H_f = np.zeros((T + 1, m, h))
    H_b = np.zeros((T + 1, m, h))
    H_f[0] = h_0
    H_b[-1] = h_t
    for f, b in zip(range(T), range(T - 1, -1, -1)):
        H_f[f + 1] = bi_cell.forward(H_f[f], X[f])
        H_b[b] = bi_cell.backward(H_b[b + 1], X[b])
    H = np.concatenate((H_f[1:], H_b[0:-1]), axis=-1)
    Y = bi_cell.output(H)
    return H, Y
