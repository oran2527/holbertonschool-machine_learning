#!/usr/bin/env python3
""" Dropout regularization """
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """ Dropout regularization """
    drop_out = {}
    A = X
    for i in range(L):
        if i > 0:
            d = (np.random.rand(A.shape[0], A.shape[1]) < keep_prob)\
                .astype(int)
            drop_out['D' + str(i)] = d
            A = np.multiply(A, d)
            A /= keep_prob
        drop_out['A' + str(i)] = A
        Z = np.dot(weights['W' + str(i + 1)], A) + weights['b' + str(i + 1)]
        # Activation
        # Softmax
        if i + 1 == L:
            exp = np.exp(Z)
            A = exp / np.sum(exp, axis=0, keepdims=True)
            drop_out['A' + str(i + 1)] = A
        else:
            # Tanh Activation
            A = np.tanh(Z)
    return drop_out
