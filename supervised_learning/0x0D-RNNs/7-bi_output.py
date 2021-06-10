#!/usr/bin/env python3
""" Bidirectional Cell Forward """
import numpy as np


class BidirectionalCell:
    """ Bidirectional Cell Forward """
    def __init__(self, i, h, o):
        """ Bidirectional Cell Forward """
        self.Whf = np.random.normal(0, 1, (i + h, h))
        self.Whb = np.random.normal(0, 1, (i + h, h))
        self.Wy = np.random.normal(0, 1, (h * 2, o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """ Bidirectional Cell Forward """
        h = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(h, self.Whf) + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """ Bidirectional Cell Forward """
        h = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(np.dot(h, self.Whb) + self.bhb)
        return h_prev

    @staticmethod
    def softmax(x):
        """ Bidirectional Cell Forward """
        return np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)

    def output(self, H):
        """ Bidirectional Cell Forward """
        T, m, h = H.shape
        Y = np.zeros((T, m, self.Wy.shape[1]))
        for t in range(T):
            y = np.dot(H[t], self.Wy) + self.by
            Y[t] = self.softmax(y)
        return Y
