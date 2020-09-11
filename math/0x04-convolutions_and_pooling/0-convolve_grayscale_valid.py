#!/usr/bin/env python3
""" 0. Valid Convolution
Function convolve_grayscale
Write a function that performs a valid convolution on grayscale images:
images is a numpy.ndarray with shape (m, h, w) containing multiple
grayscale images
m is the number of images
h is the height in pixels of the images
w is the width in pixels of the images
kernel is a numpy.ndarray with shape (kh, kw) containing the kernel
for the convolution
kh is the height of the kernel
kw is the width of the kernel
You are only allowed to use two for loops; any other loops of any kind
are not allowed
Returns: a numpy.ndarray containing the convolved images
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """ 0. Valid Convolution"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    ks1 = h - kh + 1
    cw1 = w - kw + 1
    ci = np.zeros((m, ks1, cw1))
    m_only = np.arange(0, m)
    # print(m_only)
    for row in range(ks1):
        for col in range(cw1):
            ci[m_only, row, col] = np.sum(np.multiply(
                images[m_only, row:row + kh, col:col + kw],
                kernel), axis=(1, 2))
    return(ci)
