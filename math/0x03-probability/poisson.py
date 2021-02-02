#!/usr/bin/env python3
""" program to calculate poisson distribution """


class Poisson:
    """ poisson class """

    def __init__(self, data=None, lambtha=1.):
        """ poisson class constructor """

        if data is None:
            try:
                if lambtha <= 0:
                    raise ValueError
                else:
                    self.lambtha = float(lambtha)
                self.lambtha = float(lambtha)
            except ValueError:
                print("lambtha must be a positive value")
        else:
            try:
                if type(data) != list:
                    raise TypeError
                if len(data) <= 2:
                    raise ValueError
                total = 0

                for i in data:
                    total = total + i
                self.lambtha = float(total / len(data))
            except TypeError:
                print("data must be a list")
            except ValueError:
                print("data must contain multiple values")

    def pmf(self, k):
        """function to calculate pmf distribution"""

        if type(k) != int:
            k = int(k)
        if k < 0:
            return 0
        if self.lambtha is not None:
            total = 1
            for i in range(1, k + 1):
                total = total * i
            e = 2.7182818285
            return ((e ** (- self.lambtha)) * ((self.lambtha) ** k)) / total

    def cdf(self, k):
        """function to calculate cdf distribution"""

        if type(k) != int:
            k = int(k)
        if k < 0:
            return 0
        if self.lambtha is not None:
            total = 0
            for i in range(0, k + 1):
                total = total + self.pmf(i)
            return total
