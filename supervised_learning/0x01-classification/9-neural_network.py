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
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W1 = np.random.randn(nodes, nx)

        self.__b2 = 0
        self.__A2 = 0
        self.__W2 = np.random.randn(nodes).reshape(1, nodes)

    @property
    def b1(self):
        """b1 getter"""
        return self.__b1

    @property
    def A1(self):
        """A1 getter"""
        return self.__A1

    @property
    def W1(self):
        """W1 getter"""
        return self.__W1

    @property
    def b2(self):
        """b2 getter"""
        return self.__b2

    @property
    def A2(self):
        """A2 getter"""
        return self.__A2

    @property
    def W2(self):
        """W2 getter"""
        return self.__W2
