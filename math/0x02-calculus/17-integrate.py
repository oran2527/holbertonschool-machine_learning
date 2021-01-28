#!/usr/bin/env python3
"""program to integrate a polynomial function"""


def poly_integral(poly, C=0):
    """function to calculate integral"""

    if type(poly) == list or type(C) == int:
        i = len(poly)        
        j = 1
        res = 0
        final = []
        if i > 1:            
            final.append(C)
            final.append(poly[0]) 
            while i > j: 
                res = poly[j] / (j + 1)
                if abs(res) - int(res) == 0:
                    res = int(res)                                                      
                final.append(res)
                j = j + 1
                res = 0
        elif i == 1:
            final.append(C)
            final.append(poly[0])
        else:
            return None
        return final
    return None
