#!/usr/bin/env python3
""" RNN Cell """
import numpy as np


class RNNCell:
    """ RNN Cell """
    def __init__(self, i, h, o):
        """ RNN Cell """
        self.Wh = np.random.normal(0, 1, (i + h, h))
        self.Wy = np.random.normal(0, 1, (h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def softmax(x):
        """ RNN Cell """
        return np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """ RNN Cell """
        # Concatenate hidden state and input data
        h = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(h, self.Wh) + self.bh)
        y = np.dot(h_next, self.Wy) + self.by
        return h_next, self.softmax(y)
