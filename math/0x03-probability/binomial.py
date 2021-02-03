#!/usr/bin/env python3
""" program to calculate binomial distribution """


class Binomial:
    """ binomial class """

    def __init__(self, data=None, n=1, p=0.5):
        """ binomial class constructor """

        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p >= 1 or p <= 0:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
            summ = 0
            for i in data:
                summ = summ + i
            m = summ / len(data)
            sumv = 0
            for i in data:
                sumv = sumv + ((i - m) ** 2)
            v = sumv / len(data)
            p = 1 - (v / m)
            n = round(m / p)
            p = m / n
            self.n = int(n)
            self.p = float(p)

    def pdf(self, x):
        """function to calculate pdf distribution"""

        if self.mean is not None and self.stddev is not None:
            e = 2.7182818285
            pi = 3.1415926536
            sd = self.stddev
            m = self.mean
            pwe = e ** (-1/2 * (((x - m) / sd) ** 2))
            return (1 / (sd * ((2 * pi) ** (1/2)))) * pwe

    def cdf(self, x):
        """function to calculate cdf distribution"""

        if self.mean is not None and self.stddev is not None:
            e = 2.7182818285
            pi = 3.1415926536
            sd = self.stddev
            m = self.mean
            x0 = (x - m) / (sd * (2 ** (1/2)))
            x1 = (x0 ** 3) / 3
            x2 = (x0 ** 5) / 10
            x3 = (x0 ** 7) / 42
            x4 = (x0 ** 9) / 216
            erf = (2 / (pi ** (1/2))) * (x0 - x1 + x2 - x3 + x4)
            return ((1 + erf) / 2)
