#!/usr/bin/env python3
""" convolve_grayscale_same
1. Same Convolution
function convolve grayscale same
Write a function def convolve_grayscale_same(images, kernel): that performs
a same
convolution on grayscale images:
images is a numpy.ndarray with shape (m, h, w) containing multiple grayscale
images
m is the number of images
h is the height in pixels of the images
w is the width in pixels of the images
kernel is a numpy.ndarray with shape (kh, kw) containing the kernel for the
convolution
kh is the height of the kernel
kw is the width of the kernel
if necessary, the image should be padded with 0’s
You are only allowed to use two for loops; any other loops of any kind are
not allowed
Returns: a numpy.ndarray containing the convolved images
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """ 1. Same Convolution"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    if kh % 2 == 0:
        pad_h = int((kh)/2)
        output_h = h - kh + (2*pad_h)
    else:
        pad_h = int((kh - 1)/2)
        output_h = h - kh + 1 + (2*pad_h)
    if kw % 2 == 0:
        pad_w = int((kw)/2)
        output_w = w - kw + (2*pad_w)
    else:
        pad_w = int((kw - 1)/2)
        output_w = w - kw + 1 + (2*pad_w)

    pad_images = np.pad(images,
                        pad_width=((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                        mode='constant', constant_values=0)
    image = np.arange(0, m)
    cvp_output = np.zeros((m, output_h, output_w))
    for y in range(output_h):
        for x in range(output_w):
            cvp_output[image, y, x] = \
                    (np.sum(pad_images[image,
                            y:kh + y, x:kw + x] * kernel, axis=(1, 2)))
    return cvp_output
