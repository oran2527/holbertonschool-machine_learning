#!/usr/bin/env python3
""" program to add two matrices """


def add_matrices(mat1, mat2):
    """ function to return the sum of two matrices """

    import numpy as np

    cursize1 = []
    cursize2 = []
    flag = 0
    new = []
    row = []
    finallist = []
    count = 0

    cursize1 = matrix_shape(mat1)
    cursize2 = matrix_shape(mat2)    

    if len(cursize1) == len(cursize2):                 
        for i in range(0, len(cursize1)):
            if cursize1[i] != cursize2[i]:
                flag = 1
                break
        if flag != 1:              
            new = np.hstack(np.add(mat1, mat2))
            if len(cursize1) == 1:                 
                for i in range(0, len(new)):
                    finallist.append(new[i])
                return finallist   
            if len(cursize1) == 2:                  
                for i in range(0, len(new)):
                    if count < cursize1[1]:
                        row.append(new[i])
                        count = count + 1
                    if count == cursize1[1]:
                        finallist.append(row)
                        count = 0                    
                        row = []
                return finallist
            if len(cursize1) == 4:               
                 print(new)
                      
    else:
        return None


def matrix_shape(matrix):
    """ matrix to define a matrix """

    import numpy as np

    x = np.array(matrix)
    return x.shape 
