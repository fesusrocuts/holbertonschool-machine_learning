#!/usr/bin/env python3
""" 5. Multiple Kernels
function convolve channels and multiples kernels
Write a function  that performs a convolution on images using multiple kernels:
images is a numpy.ndarray with shape (m, h, w, c) containing multiple images
m is the number of images
h is the height in pixels of the images
w is the width in pixels of the images
c is the number of channels in the image
kernels is a numpy.ndarray with shape (kh, kw, c, nc) containing the kernels
for the convolution
kh is the height of a kernel
kw is the width of a kernel
nc is the number of kernels
padding is either a tuple of (ph, pw), ‘same’, or ‘valid’
if ‘same’, performs a same convolution
if ‘valid’, performs a valid convolution
if a tuple:
ph is the padding for the height of the image
pw is the padding for the width of the image
the image should be padded with 0’s
stride is a tuple of (sh, sw)
sh is the stride for the height of the image
sw is the stride for the width of the image
You are only allowed to use three for loops; any other loops
of any kind are not allowed
Returns: a numpy.ndarray containing the convolved images
"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """ 5. Multiple Kernels"""
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    sh = stride[0]
    sw = stride[1]
    if isinstance(padding, tuple):
        ph = padding[0]
        pw = padding[1]
    elif padding == 'same':
        ph, pw = int((kh-1)/2), int((kw-1)/2)
    else:
        ph, pw = 0, 0
    new_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        mode='constant', constant_values=0)
    m, new_h, new_w, c = new_images.shape
    ch = int(np.floor(((h - kh + 2*ph) / sh) + 1))
    cw = int(np.floor(((w - kw + 2*pw) / sw) + 1))
    ci = np.zeros((m, ch, cw, nc))
    m_only = np.arange(0, m)
    ch_only = np.arange(0, c)
    for row in range(ch):
        for col in range(cw):
            for n_k in range(nc):
                a = row*sh
                b = row*sh + kh
                c = col*sw
                d = col*sw + kw
                prevmult = np.multiply(
                    new_images[m_only, a:b, c:d, ], kernels[n_k])
                ci[m_only, row, col, n_k] = np.sum(prevmult, axis=(1, 2, 3))
    return(ci)
