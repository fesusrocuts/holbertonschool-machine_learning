#!/usr/bin/env python3
""" fn performs backProp over a convolutional layer
"""

import numpy as np


def conv_backward2(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
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
    # retrive dimensions
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh = stride[0]
    sw = stride[1]

    # calculate padding for 'valid' and 'same'
    if padding == 'valid':
        ph = 0
        pw = 0
    if padding == 'same':
        ph = np.ceil(((sh * h_prev) - sh + kh - h_prev) / 2)
        ph = int(ph)
        pw = np.ceil(((sw * w_prev) - sw + kw - w_prev) / 2)
        pw = int(pw)

    # initializing output: dA_prev, dW, db
    dA_prev = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)

    # pad previous images
    padA_prev = np.pad(A_prev, [(0, 0), (ph, ph), (pw, pw), (0, 0)],
                       mode='constant', constant_values=0)
    paddA_prev = np.pad(dA_prev, [(0, 0), (ph, ph), (pw, pw), (0, 0)],
                        mode='constant', constant_values=0)

    # vectorize
    imagesVector = np.arange(0, m)

    # Looping engine: loop over m, h_new, w_new, c_new
    for n in range(m):
        # select training example i
        a_padA_prev = padA_prev[n, :, :, :]
        da_paddA_prev = paddA_prev[n, :, :, :]

        for i in range(h_new):
            for j in range(w_new):
                for k in range(c_new):
                    # finding the corners
                    istart = i * sh
                    iend = istart + kh
                    jstart = j * sw
                    jend = jstart + kw

                    # slice from previous layer
                    aSlice = a_padA_prev[istart:iend, jstart:jend, :]

                    # update gradients
                    da_paddA_prev[
                        istart:iend, jstart:jend, :] += W[:, :, :, k] * dZ[
                            n, i, j, k]

                    dW[:, :, :, k] += aSlice * dZ[n, i, j, k]
                    db[:, :, :, k] += dZ[n, i, j, k]
    assert(dA_prev.shape == A_prev.shape)
    return dA_prev, dW, db





def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    2. Convolutional Back Prop
    dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
               numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    dW -- gradient of the cost with respect to the weights of the conv layer (W)
          numpy array of shape (f, f, n_C_prev, n_C)
    db -- gradient of the cost with respect to the biases of the conv layer (b)
          numpy array of shape (1, 1, 1, n_C)
    """
    # Retrieve information from "cache"
    (A_prev, W, b, hparameters) = cache

    # Retrieve dimensions from A_prev
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # Retrieve dimensions from W
    (f, f, n_C_prev, n_C) = W.shape

    # Retrieve dimensions from dZ
    (m, n_H, n_W, n_C) = dZ.shape

    dA_prev = np.random.randn(m, n_H_prev, n_W_prev, n_C_prev)
    dW = np.random.randn(f, f, n_C_prev, n_C)
    db = np.random.randn(f, f, n_C_prev, n_C)

    stride = 1
    pad = 0
    if padding is 'same':
        pad = W.shape[0]

    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)

    for i in range(m):

        a_prev_pad = A_prev_pad[i, :, :, :]
        da_prev_pad = dA_prev_pad[i, :, :, :]

        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):

                    vert_start = stride * h
                    vert_end = stride * h + f
                    horiz_start = stride * w
                    horiz_end = stride * w + f

                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]

        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]

    return dA_prev, dW, db
