#!/usr/bin/env python3
"""program to derivate a polynomial function"""


def poly_derivative(poly):
    """function to calculate derivate"""

    if type(poly) == list:
        i = len(poly)
        final = []
        if i > 1:
            while i > 1:
                i = i - 1
                final.append(poly[i] * (i))
        elif i == 1:
            final.append(0)
        else:
            return None
        return final
    return None
