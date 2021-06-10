#!/usr/bin/env python3
""" LSTM Cell """
import numpy as np


class LSTMCell:
    """ LSTM Cell """
    def __init__(self, i, h, o):
        """ LSTM Cell """
        self.Wf = np.random.normal(0, 1, (i + h, h))
        self.Wu = np.random.normal(0, 1, (i + h, h))
        self.Wc = np.random.normal(0, 1, (i + h, h))
        self.Wo = np.random.normal(0, 1, (i + h, h))
        self.Wy = np.random.normal(0, 1, (h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def sigmoid(x):
        """ LSTM Cell """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        """ LSTM Cell """
        return np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)

    def forward(self, h_prev, c_prev, x_t):
        """ LSTM Cell """
        h = np.concatenate((h_prev, x_t), axis=1)

        f_t = self.sigmoid(np.dot(h, self.Wf) + self.bf)
        i_t = self.sigmoid(np.dot(h, self.Wu) + self.bu)
        c_t = np.tanh(np.dot(h, self.Wc) + self.bc)
        o_t = self.sigmoid(np.dot(h, self.Wo) + self.bo)
        c = f_t * c_prev + i_t * c_t
        h_next = o_t * np.tanh(c)
        y = np.dot(h_next, self.Wy) + self.by

        return h_next, c, self.softmax(y)
