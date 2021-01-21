#!/usr/bin/env python3
""" matrix multiplication """


def mat_mul(mat1, mat2):
    """ function to multiply two matrices """

    cursize1 = []
    cursize2 = []
    newmat = []
    add = 0
    finalmat = []

    cursize1 = matrix_shape(mat1)
    cursize2 = matrix_shape(mat2)

    if cursize1[1] == cursize2[0]:
        for i in range(0, len(mat1)):
            for w in range(0, len(mat2[0])):
                for m in range(0, len(mat2)):
                    add = add + mat1[i][m] * mat2[m][w]
                newmat.append(add)
                add = 0
            finalmat.append(newmat)
            newmat = []
        return finalmat
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
