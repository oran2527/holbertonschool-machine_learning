#!/usr/bin/env python3
""" Dropout with gradient descent """
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """ Dropout with gradient descent """
    m = Y.shape[1]
    # cost
    dz = cache['A' + str(L)] - Y
    for i in range(L, 0, -1):
        a = 'A' + str(i - 1)
        w = 'W' + str(i)
        b = 'b' + str(i)
        A = cache[a]
        dw = (1 / m) * np.dot(dz, np.transpose(A))
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        if 'D' + str(i - 1) in cache:
            mask = cache['D' + str(i - 1)]
            dz = np.matmul(np.transpose(weights[w]), dz) \
                * (1 - (A**2)) * (mask / keep_prob)
        else:
            dz = np.matmul(np.transpose(weights[w]), dz) * A * (1 - A)
        weights[w] = weights[w] - (alpha * dw)
        weights[b] = weights[b] - (alpha * db)
