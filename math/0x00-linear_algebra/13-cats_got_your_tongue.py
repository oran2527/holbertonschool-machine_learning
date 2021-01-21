#!/usr/bin/env python3
""" program to concatenate two matrices along specific axis """
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """ function to return a new array from two matrices along an axis """

    return np.concatenate((mat1, mat2), axis)
