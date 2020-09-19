#!/usr/bin/env python3
""" fn performs backProp over a convolutional layer
"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
     vn layer
     dZ: np.ndarray, partial derivatives
        m: num examples
        h_new: output height
        w_new: output weight
        c_new: output channels
    A_prev: np.ndarray, prev output layer
        m: num examples
        h_prev: prev height
        w_prev: prev width
        c_prev: prev channels
    w: np.ndarray, kernels
        kh: filter height
        kw: filter width
        c_prev: prev channels
        c_new: output channels
     b: (1, 1, 1, c_new)
     padding: 'same' or 'valid'
     stride: (sh, sw)
     Return: dA_prev, dW, db
     """
    # Number of samples
    m = A_prev.shape[0]

    # Create placeholders for derivatives
    dA_prev = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)

    # Size of stride from forward prop
    s = stride[0]

    # Retrieve height and width of kernel
    kh, kw = W.shape[0], W.shape[1]

    # Height as h, width as w, and depth of dZ
    h_new, w_new, c_new = dZ.shape[1], dZ.shape[2], dZ.shape[3]

    # Retrieve height, width, and depth of previous input layer
    h_prev, w_prev, c_prev = A_prev.shape[1], A_prev.shape[2], A_prev.shape[3]

    # Padding
    p = (kh + s * (h_prev - 1) - h_prev) // 2

    if padding == "same":
        p = (h_prev - kh + 2 * p) // s + 1
        prev1 = ((0, 0), (p, p), (p, p), (0, 0))
        dA_prev = np.pad(dA_prev, prev1, 'constant', constant_values=0)
        A_prev_pad = np.pad(A_prev, prev1, 'constant', constant_values=0)
    else:
        p = 0
        A_prev_pad = A_prev

    for sample in range(m):
        for depth in range(c_new):
            for h in range(h_new):
                for w in range(w_new):
                    dA_prev[sample, h*s:h*s+kh, w*s:w*s+kw, :] +=\
                        W[:, :, :, depth] * dZ[sample, h, w, depth]
                    dW[:, :, :, depth] +=\
                        A_prev_pad[sample, h*s:h*s+kh, w*s:w*s+kw, :] * \
                        dZ[sample, h, w, depth]
                    db[:, :, :, depth] += dZ[sample, h, w, depth]

    if p:
        return dA_prev[:, p:-p, p:-p, :], dW, db
    return dA_prev, dW, db
