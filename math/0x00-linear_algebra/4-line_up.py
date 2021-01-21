#!/usr/bin/env python3
""" array adder """


def add_arrays(arr1, arr2):
    """ function to add two arrays """

    cursize1 = []
    cursize2 = []
    finallist = []
    flag = 0

    cursize1 = matrix_shape(arr1)
    cursize2 = matrix_shape(arr2)

    if len(cursize1) == len(cursize2):
        for i in range(0, len(cursize1)):
            if cursize1[i] != cursize2[i]:
                flag = 1
                break
        if flag != 1:
            for i in range(0, len(arr1)):
                finallist.append(arr1[i] + arr2[i])
            return finallist
    else:
        return None


def matrix_shape(matrix_sub):
    """ matrix to define a matrix """

    size = []
    try:
        size.append(len(matrix_sub))
        size.append(len(matrix_sub[0]))
        size.append(len(matrix_sub[0][0]))
        return size
    except Exception as ex:
        return size
