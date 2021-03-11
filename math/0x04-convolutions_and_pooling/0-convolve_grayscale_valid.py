#!/usr/bin/env python3
""" Convolution on grayscale """
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """ Convolution on grayscale """
    # Input dimensions
    m = images.shape[0]
    ih = images.shape[1]
    iw = images.shape[2]
    # Kernel dimension
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    # Output image dimensions
    nh = (ih - kh) + 1
    nw = (iw - kw) + 1
    new_img = np.zeros((m, nh, nw))
    x = y = 0
    while y < nh:
        op_kernel = images[:, y:y+kh, x:x+kw] * kernel
        new_img[:, y, x] = np.sum(np.sum(op_kernel, axis=1), axis=1)
        if x + 1 == nw:
            x = 0
            y += 1
        else:
            x += 1
    return new_img
