#!/usr/bin/env python3
""" program to calculate exponential distribution """


class Exponential:
    """ exponential class """

    def __init__(self, data=None, lambtha=1.):
        """ exponential class constructor """

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
                self.lambtha = float(1/(total / len(data)))
            except TypeError:
                print("data must be a list")
            except ValueError:
                print("data must contain multiple values")

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
            return 1 - (e ** ( -1 * self.lambtha * x))       
