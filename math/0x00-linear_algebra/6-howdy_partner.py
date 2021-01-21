#!/usr/bin/env python3
""" program to concatenate two arrays """


def cat_arrays(arr1, arr2):
    """ function to concatenate """

    finalarr = []
    for i in arr1:
        finalarr.append(i)
    for i in arr2:
        finalarr.append(i)
    return finalarr
