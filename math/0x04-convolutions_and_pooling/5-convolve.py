#!/usr/bin/env python3
""" Convolution with multiple kernels """
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """ Convolution with multiple kernels """
    # Input dimensions
    m = images.shape[0]
    ih = images.shape[1]
    iw = images.shape[2]
    c = images.shape[3]
    # Kernel dimensions
    kh = kernels.shape[0]
    kw = kernels.shape[1]
    nc = kernels.shape[3]
    # Stride dimensions
    sh = stride[0]
    sw = stride[1]
    if type(padding) == tuple:
        ph = padding[0]
        pw = padding[1]
    elif padding == 'same':
        ph = int((((ih - 1) * sh - ih + kh) / 2)) + 1
        pw = int((((iw - 1) * sw - iw + kw) / 2)) + 1
    elif padding == 'valid':
        ph = 0
        pw = 0
    pad_images = np.pad(images, ((0,), (ph,), (pw,), (0,)), 'constant')
    nh = int(((ih + (2 * ph) - kh) / sh)) + 1
    nw = int(((iw + (2 * pw) - kw) / sw)) + 1
    new_img = np.zeros((m, nh, nw, nc))
    x = y = 0
    i = j = 0
    while j < nh:
        k = 0
        while k < nc:
            op_kernel = (pad_images[:, y:y+kh, x:x+kw, :]
                         * kernels[:, :, :, k])
            new_img[:, j, i, k] = op_kernel.sum(axis=(1, 2, 3))
            k += 1
        if i + 1 >= nw:
            x = 0
            i = 0
            y += sh
            j += 1
        else:
            x += sw
            i += 1
    return new_img
