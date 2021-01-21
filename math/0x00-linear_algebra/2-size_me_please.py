#!/usr/bin/env python3
""" matrix shape calculator """


def matrix_shape(matrix):
    """ matrix to define a matrix """

    size = []
    try:
        if type(matrix) == list:
            size.append(int(len(matrix)))
        if type(matrix[0]) == list:
            size.append(int(len(matrix[0])))
        if type(matrix[0][0]) == list:
            size.append(int(len(matrix[0][0])))
        return size
    except Exception as ex:
        return size
