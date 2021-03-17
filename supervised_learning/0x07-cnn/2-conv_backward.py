#!/usr/bin/env python3
""" Convolutional backpropagation """
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """ Convolutional backpropagation """
    # Get dimensions
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    # Stride dimensions
    sh = stride[0]
    sw = stride[1]
    if type(padding) == tuple:
        ph, pw = padding.shape
    elif padding == 'same':
        ph = int((((h_prev - 1) * sh - h_prev + kh) / 2)) + 1
        pw = int((((w_prev - 1) * sw - w_prev + kw) / 2)) + 1
    elif padding == 'valid':
        ph = 0
        pw = 0
    # Initialize dA_prev, dW, db with the correct shapes
    dA_prev = np.zeros((A_prev.shape))
    dW = np.zeros((W.shape))
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    # Pad A_prev and dA_prev
    A_prev_pad = np.pad(A_prev, ((0,), (ph,), (pw,), (0,)), 'constant')
    dA_prev_pad = np.pad(dA_prev, ((0,), (ph,), (pw,), (0,)), 'constant')

    for i in range(m):
        # Select example
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    # Variables to define slice size
                    h_start = h * sh
                    h_end = h_start + kh
                    w_start = w * sw
                    w_end = w_start + kw

                    # Slice a_prev_pad
                    a_slice = a_prev_pad[h_start:h_end, w_start:w_end, :]

                    # Update gradients for the window and the filter's
                    da_prev_pad[h_start:h_end,
                                w_start:w_end, :] += \
                        W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]

        # Unpad dA
        if padding == 'same':
            dA_prev[i, :, :, :] += da_prev_pad[ph:-ph, pw:-pw, :]
        if padding == 'valid':
            dA_prev[i, :, :, :] += da_prev_pad

    return dA_prev, dW, db
