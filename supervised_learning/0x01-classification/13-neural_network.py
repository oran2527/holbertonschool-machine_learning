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

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""

        XT = np.transpose(X)
        W1T = np.transpose(self.W1)
        A_1 = np.transpose(np.matmul(XT, W1T)) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-1 * A_1))

        A1T = np.transpose(self.A1)
        W2T = np.transpose(self.W2)
        A_2 = np.transpose(np.matmul(A1T, W2T)) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-1 * A_2))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """cost of the model"""

        m = np.shape(Y)

        j1 = -1 * (1 / m[1])
        j3 = np.multiply(Y, np.log(A))
        j4 = np.multiply(1 - Y, np.log(1.0000001 - A))
        j = j1 * np.sum(j3 + j4)
        return j

    def evaluate(self, X, Y):
        """Evaluates the neuronal network’s predictions"""

        self.forward_prop(X)
        j = self.cost(Y, self.__A2)
        p = np.where(self.__A2 >= 0.5, 1, 0)
        return p, j

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron"""

        m = (np.shape(X))[1]

        h_theta_minus_y_1 = np.subtract(A1, Y)
        trans_X = np.transpose(X)
        sigma_g_1 = np.matmul(h_theta_minus_y_1, trans_X)
        j_theta_1 = np.multiply((1 / m), sigma_g_1)
        learning_rate_j_theta_1 = np.multiply(alpha, j_theta_1)
        gradient_1 = np.subtract(self.W1, learning_rate_j_theta_1)

        self.__W1 = gradient_1
