#!/usr/bin/env python3
""" program to calculate normal distribution """


class Normal:
    """ normal class """

    def __init__(self, data=None, mean=0., stddev=1.):
        """ normal class constructor """

        if data is None:
            try:
                if stddev <= 0:
                    raise ValueError
                self.mean = float(mean)
                self.stddev = float(stddev)
            except ValueError:
                print("stddev must be a positive value")
        else:
            try:
                if type(data) != list:
                    raise TypeError
                if len(data) <= 2:
                    raise ValueError
                total = 0
                for i in data:
                    total = total + i
                self.mean = float(total / len(data))
                sigma = 0
                for i in data:
                    sigma = sigma + (i - self.mean) ** 2
                self.stddev = (sigma / len(data)) ** (1/2)
            except TypeError:
                print("data must be a list")
            except ValueError:
                print("data must contain multiple values")

    def z_score(self, x):
        """function to calculate z_score"""

        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """function to calculate x_value"""

        return z * self.stddev + self.mean

    def pdf(self, x):
        """function to calculate pdf distribution"""

        if x < 0:
            return 0
        if self.lambtha is not None:
            e = 2.7182818285
            return self.lambtha * (e ** (-1 * self.lambtha * x))

    def cdf(self, x):
        """function to calculate cdf distribution"""

        if x < 0:
            return 0
        if self.lambtha is not None:
            e = 2.7182818285
            return 1 - (e ** (-1 * self.lambtha * x))
