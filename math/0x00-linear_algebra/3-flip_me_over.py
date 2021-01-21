#!/usr/bin/env python3
""" matrix transposer """


def matrix_transpose(matrix):
    """ function to transpose a matrix """

    newmat = []
    newmat2 = []
    finalmat = []
    cursize = []

    cursize = matrix_shape(matrix)
    for i in range(0, cursize[0]):
        for j in range(0, cursize[1]):
            newmat.append(matrix[i][j])
    for i in range(0, cursize[1]):
        newmat2.append(newmat[i])
        j = i
        while(len(newmat2) < cursize[0]):
            if j < len(newmat):
                newmat2.append(newmat[j + cursize[1]])
                j = j + cursize[1]
        finalmat.append(newmat2)
        newmat2 = []
    return finalmat


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
