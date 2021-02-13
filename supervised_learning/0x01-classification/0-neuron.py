#!/usr/bin/env python3
""" Neuron Class """

import numpy as np


class Neuron:
    """ Class Neuron """

    def __init__(self, nx):
        """ Neuron initializer """

        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.b = 0
        self.A = 0
        self.W = np.random.randn(nx).reshape(1, nx)
