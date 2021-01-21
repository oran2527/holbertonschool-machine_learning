#!/usr/bin/env python3
""" matrix shape calculator """


def matrix_shape(matrix):
    """ matrix to define a matrix """

    size = []
    try:
        size.append(len(matrix))
        size.append(len(matrix[0]))
        size.append(len(matrix[0][0]))
        return size
    except Exception as ex:
        return size
