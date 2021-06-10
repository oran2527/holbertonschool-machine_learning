#!/usr/bin/env python3
""" GRU Cell """
import numpy as np


class GRUCell:
    """ GRU Cell """
    def __init__(self, i, h, o):
        """ GRU Cell """
        self.Wz = np.random.normal(0, 1, (i + h, h))
        self.Wr = np.random.normal(0, 1, (i + h, h))
        self.Wh = np.random.normal(0, 1, (i + h, h))
        self.Wy = np.random.normal(0, 1, (h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def sigmoid(x):
        """ GRU Cell """
        return 1/(1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        """ GRU Cell """
        return np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """ GRU Cell """
        h = np.concatenate((h_prev, x_t), axis=1)
        z_t = self.sigmoid(np.dot(h, self.Wz) + self.bz)
        r_t = self.sigmoid(np.dot(h, self.Wr) + self.br)
        h_t = np.concatenate((r_t * h_prev, x_t), axis=1)
        h_t = np.tanh(np.dot(h_t, self.Wh) + self.bh)
        h_next = (1 - z_t) * h_prev + z_t * h_t
        y = np.dot(h_next, self.Wy) + self.by
        return h_next, self.softmax(y)
