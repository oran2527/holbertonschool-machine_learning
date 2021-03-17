#!/usr/bin/env python3
""" Convolution layer with pooling """
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """ Convolution layer with pooling """
    # Define variables
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh = kernel_shape[0]
    kw = kernel_shape[1]
    # Stride dimensions
    sh = stride[0]
    sw = stride[1]
    # Output dimensions
    nh = int(((h_prev - kh) / sh)) + 1
    nw = int(((w_prev - kw) / sw)) + 1
    new_img = np.zeros((m, nh, nw, c_prev))
    x = y = 0
    i = j = 0
    while j < nh:
        if mode == 'max':
            op_kernel = np.max(A_prev[:, y:y+kh, x:x+kw, :],
                               axis=(1, 2))
        elif mode == 'avg':
            op_kernel = np.average(A_prev[:, y:y+kh, x:x+kw, :],
                                   axis=(1, 2))
        new_img[:, j, i] = op_kernel
        if i + 1 >= nw:
            x = 0
            i = 0
            y += sh
            j += 1
        else:
            x += sw
            i += 1
    return new_img
