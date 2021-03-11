#!/usr/bin/env python3
""" Convolution on gray scale 'SAME'"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """ Convolution on gray scale 'SAME'"""
    # Input dimensions
    m = images.shape[0]
    ih = images.shape[1]
    iw = images.shape[2]
    # Kernel dimension
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    # Padding
    ph = int(kh / 2)
    pw = int(kw / 2)
    # Output dimensions
    nh = ih
    nw = iw
    new_img = np.ndarray((m, nh, nw))
    # Padding all images
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
