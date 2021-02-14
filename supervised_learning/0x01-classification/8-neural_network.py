#!/usr/bin/env python3
""" NeuralNetwork Class """

import numpy as np


class NeuralNetwork:
    """ Class NeuralNetwork """

    def __init__(self, nx, nodes):
        """ NeuralNetwork initializer """

        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nx < 1:
            raise ValueError("nodes must be a positive integer")

        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        self.W1 = np.random.randn(nodes, nx)

        self.b2 = 0
        self.A2 = 0
        self.W2 = np.random.randn(nodes).reshape(1, nodes)
