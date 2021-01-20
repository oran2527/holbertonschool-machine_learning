#!/usr/bin/env python3
""" matrix 2D adder """

def add_matrices2D(mat1, mat2):
    """ function to add two matrices """

    cursize1 = []
    cursize2 = []
    newmat = []
    finallist = []
    flag = 0

    cursize1 = matrix_shape(mat1)
    cursize2 = matrix_shape(mat2)

    if len(cursize1) == len(cursize2):
        for i in range(0, len(cursize1)):
            if cursize1[i] != cursize2[i]:
                flag = 1
                break
        if flag != 1:
            for i in range(0, len(mat1)):
                for j in range(0, len(mat1[i])):
                    newmat.append(mat1[i][j] + mat2[i][j])
                finallist.append(newmat)
                newmat = []
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
    except:
        return size
