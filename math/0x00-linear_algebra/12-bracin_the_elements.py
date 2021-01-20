#!/usr/bin/env python3
""" program to calculate the basic operations of two matrices """


def np_elementwise(mat1, mat2):
    """ function to return the basic operations of twoa matrices """

    import numpy as np

    return mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2
