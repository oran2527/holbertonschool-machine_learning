#!/usr/bin/env python3
""" matrix 2D concatenater """

def cat_matrices2D(mat1, mat2, axis=0):
    """ function to concatenate two 2d matrices """

    cursize1 = []
    cursize2 = []
    newmat = []
    finallist = []
       
    
    cursize1 = matrix_shape(mat1)
    cursize2 = matrix_shape(mat2)    
        
    if axis == 0:
        for i in mat1:
            finallist.append(i.copy())
        for i in mat2:
            finallist.append(i.copy())
        return finallist
    else:
        if axis > 0:
            for i in mat1:
                newmat.append(i.copy())            
            for i in range(0, len(newmat)):
                for j in range(0, len(mat2[i])):
                    newmat[i].append(mat2[i][j])             
            return newmat                            
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
