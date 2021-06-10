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
