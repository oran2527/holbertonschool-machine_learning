#!/usr/bin/env python3
"""program to generate sigma sum"""


def summation_i_squared(n):
    """sum of a square variable in a sigma"""

    if type(n) != int or n < 1:
        return None
    else:
        if type(n) == int:
            if n > 1:
                return n**2 + summation_i_squared(n-1)
            return n
