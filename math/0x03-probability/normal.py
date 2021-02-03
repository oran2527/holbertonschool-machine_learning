#!/usr/bin/env python3
""" program to calculate normal distribution """


class Normal:
    """ normal class """

    def __init__(self, data=None, mean=0., stddev=1.):
        """ normal class constructor """

        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)

        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
            total = 0
            for i in data:
                total = total + i
            self.mean = float(total / len(data))
            sigma = 0
            for i in data:
                sigma = sigma + (i - self.mean) ** 2
            self.stddev = (sigma / len(data)) ** (1/2)

    def z_score(self, x):
        """function to calculate z_score"""

        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """function to calculate x_value"""

        return z * self.stddev + self.mean

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
