#!/usr/bin/env python3
""" Gaussian Process """
import numpy as np


class GaussianProcess():
    """ Gaussian Process """
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """ Gaussian Process """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """ Gaussian Process """
        sqdist = np.sum(X1**2, 1).reshape(-1, 1)\
            + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)
