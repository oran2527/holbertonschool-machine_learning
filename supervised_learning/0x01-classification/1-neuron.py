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
            raise ValueError("nx must be an integer")

        self.__b = 0
        self.__A = 0
        self.__W = np.random.randn(nx).reshape(1, nx)

    @property
    def b(self):
        """getter for b attribute"""
        return self.__b

    @property
    def A(self):
        """getter for A attribute"""
        return self.__A

    @property
    def W(self):
        """getter for W attribute"""
        return self.__W
