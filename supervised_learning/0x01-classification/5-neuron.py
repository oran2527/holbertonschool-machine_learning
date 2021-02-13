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

    def forward_prop(self, X):
        """forward propagation function"""
        e = 2.7182818285

        fp = np.matmul(self.W, X) + self.b
        fsig = 1 / (1 + (e ** -(fp)))
        self.__A = fsig
        return self.__A

    def cost(self, Y, A):
        """cost of the model using logistic regression"""

        m = np.shape(Y)

        j1 = -1 * (1 / m[1])
        j3 = np.multiply(Y, np.log(A))
        j4 = np.multiply(1 - Y, np.log(1.0000001 - A))
        j = j1 * np.sum(j3 + j4)
        return j

    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions"""

        j = self.cost(Y, self.forward_prop(X))
        p = np.where(self.forward_prop(X) < 0.5, 0, 1)
        return p, j

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron"""

        m = (np.shape(X))[1]

        h_theta_minus_y = np.subtract(A, Y)
        trans_X = np.transpose(X)
        sigma_g = np.matmul(h_theta_minus_y, trans_X)
        j_theta = np.multiply((1 / m), sigma_g)
        learning_rate_j_theta = np.multiply(alpha, j_theta)
        gradient = np.subtract(self.W, learning_rate_j_theta)

        sigma_b = np.sum(h_theta_minus_y)
        gradient_bias = (1 / m) * (alpha * sigma_b)

        self.__W = gradient
        self.__b = gradient_bias
