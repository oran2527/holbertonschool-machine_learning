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

    def forward_prop(self, X):
        """forward propagation function"""

        fp = np.matmul(self.W, X) + self.b
        fsig = 1 / (1 + np.exp(-1 * fp))
        self.__A = fsig
        return self.__A

    def cost(self, Y, A):
        """cost of the model using logistic regression"""

        m = Y.shape[1]

        j = - (1 / m) * np.sum(
            np.multiply(
                Y, np.log(A)) + np.multiply(
                    1 - Y, np.log(1.0000001 - A)))

        return j

    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions"""

        self.forward_prop(X)
        j = self.cost(Y, self.__A)
        p = np.where(self.__A >= 0.5, 1, 0)
        return p, j
