#!/usr/bin/env python3
"""
fn build a projection block
"""


import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """
    A_prev is the output from the previous layer
    filters is a tuple or list containing F11, F3, F12, respectively:
        F11 is the number of filters in the first 1x1 convolution
        F3 is the number of filters in the 3x3 convolution
        F12 is the number of filters in the second 1x1 convolution
            as well as the 1x1 convolution in the shortcut connection
    s is the stride of the first convolution in both the main path
        and the shortcut connection
    All convolutions inside the block should be followed by
        batch normalization along the channels axis and a
        rectified linear activation (ReLU), respectively.
    All weights should use he normal initialization
    Returns: the activated output of the projection block
    """
    init = K.initializers.he_normal(seed=None)
    r = K.layers.Conv2D(filters[0], 1, s,
                        kernel_initializer=init)(A_prev)
    r = K.layers.BatchNormalization()(r)
    r = K.layers.Activation('relu')(r)
    r = K.layers.Conv2D(filters[1], 3, padding='same',
                        kernel_initializer=init)(r)
    r = K.layers.BatchNormalization()(r)
    r = K.layers.Activation('relu')(r)
    r = K.layers.Conv2D(filters[2], 1,
                        kernel_initializer=init)(r)
    r = K.layers.BatchNormalization()(r)
    r1 = K.layers.Conv2D(filters[2], 1, s,
                         kernel_initializer=init)(A_prev)
    r1 = K.layers.BatchNormalization()(r1)
    r = K.layers.add([r, r1])
    return K.layers.Activation('relu')(r)
