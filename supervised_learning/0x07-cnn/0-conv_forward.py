#!/usr/bin/env python3
""" Convolution of a layer """
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same",
                 stride=(1, 1)):
    """ Convolution of a layer """
    # Define variables
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    # Stride dimensions
    sh = stride[0]
    sw = stride[1]
    # Set padding
    if padding == 'same':
        ph = int((((h_prev - 1) * sh - h_prev + kh) / 2))
        pw = int((((w_prev - 1) * sw - w_prev + kw) / 2))
    elif padding == 'valid':
        ph = 0
        pw = 0

    # Output dimensions
    nh = int(((h_prev + (2 * ph) - kh) / sh)) + 1
    nw = int(((w_prev + (2 * pw) - kw) / sw)) + 1

    pad_images = np.pad(A_prev, ((0,), (ph,), (pw,), (0,)), 'constant')
    new_img = np.zeros((m, nh, nw, c_new))
    x = y = 0
    i = j = 0
    while j < nh:
        k = 0
        while k < c_new:
            op_filter = (pad_images[:, y:y+kh, x:x+kw, :]
                         * W[:, :, :, k])
            new_img[:, j, i, k] = op_filter.sum(axis=(1, 2, 3))
            k += 1
        if i + 1 >= nw:
            x = 0
            i = 0
            y += sh
            j += 1
        else:
            x += sw
            i += 1
    new_img += b
    return activation(new_img)
