#!/usr/bin/env python3
""" Bayesian Optimization """
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization():
    """ Bayesian Optimization """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """ Bayesian Optimization """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """ Bayesian Optimization """
        mu, sigma = self.gp.predict(self.X_s)
        if not self.minimize:
            X_next = np.amax(self.gp.Y)
            imp = (mu - X_next - self.xsi)
        if self.minimize:
            X_next = np.amin(self.gp.Y)
            imp = (X_next - mu - self.xsi)
        n = len(sigma)
        Z = [imp[i] / sigma[i] if sigma[i] > 0 else 0 for i in range(n)]
        EI = np.zeros(sigma.shape)
        for i in range(n):
            if sigma[i] > 0:
                EI[i] = imp[i] * norm.cdf(Z[i]) + sigma[i] * norm.pdf(Z[i])
        return self.X_s[np.argmax(EI)], EI

    def optimize(self, iterations=100):
        """ Bayesian Optimization """
        sampled = []
        for _ in range(iterations):
            X_next, _ = self.acquisition()
            if X_next in sampled:
                break
            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)
            sampled.append(X_next)

        if not self.minimize:
            idx = np.argmax(self.gp.Y)
        if self.minimize:
            idx = np.argmin(self.gp.Y)

        return self.gp.X[idx], self.gp.Y[idx]

    @staticmethod
    def f(x):
        """ Bayesian Optimization """
        return np.sin(5 * x) + 2 * np.sin(-2 * x)
