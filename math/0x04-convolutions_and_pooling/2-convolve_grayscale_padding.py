#!/usr/bin/env python3
""" Convultion with padding """
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """ Convultion with padding """
    # Input dimensions
    m = images.shape[0]
    ih = images.shape[1]
    iw = images.shape[2]
    # Kernel dimension
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    # padding dimensions
    ph = padding[0]
    pw = padding[1]
    # new image dimensions
    nh = (ih - kh) + (2 * ph) + 1
    nw = (iw - kw) + (2 * pw) + 1
    new_img = np.ndarray((m, nh, nw))
    pad_images = np.pad(images, ((0,), (ph,), (pw,)), 'constant')
    x = y = 0
    while y < nh:
        op_kernel = pad_images[:, y:y+kh, x:x+kw] * kernel
        new_img[:, y, x] = np.sum(np.sum(op_kernel, axis=1), axis=1)
        if x + 1 == nw:
            x = 0
            y += 1
        else:
            x += 1
    return new_img
