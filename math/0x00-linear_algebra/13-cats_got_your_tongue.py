#!/usr/bin/env python3
""" program to concatenate two matrices along specific axis """


def np_cat(mat1, mat2, axis=0):
    """ function to return a new numpy.ndarray from two matrices along a specific axis """

    import numpy as np

    return np.concatenate((mat1, mat2), axis)
