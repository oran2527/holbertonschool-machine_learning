#!/usr/bin/env python3
""" Determinant of a matrix """


def determinant(matrix):
    """ Determinant of a matrix """
    if type(matrix) is not list or not matrix:
        raise TypeError("matrix must be a list of lists")
    for x in matrix:
        if type(x) is not list:
            raise TypeError("matrix must be a list of lists")
    if len(matrix) > 0 and len(matrix[0]) > 0:
        if len(matrix) != len(matrix[0]):
            raise ValueError("matrix must be a square matrix")

    if len(matrix[0]) == 0:
        return 1
    if len(matrix[0]) == 1:
        return matrix[0][0]
    det = 0
    if len(matrix) == 2:
        det = matrix[0][0] * matrix[1][1]
        det -= matrix[0][1] * matrix[1][0]
        return det
    sign = 1
    for j in range(len(matrix[0])):
        tmp = matrix[1:]
        for i in range(len(tmp)):
            tmp[i] = tmp[i][0:j] + tmp[i][j + 1:]
        det += (sign * matrix[0][j]) * determinant(tmp)
        sign *= -1
    return det
