#!/usr/bin/env python3
"""
fn builds inception block
"""

import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    A_prev is the output from the previous layer
    filters is a tuple or list containing F1, F3R,
        F3,F5R, F5, FPP, respectively:
            F1 is the number of filters in the 1x1 convolution
            F3R is the number of filters in the 1x1 convolution before
                the 3x3 convolution
            F3 is the number of filters in the 3x3 convolution
            F5R is the number of filters in the 1x1 convolution before
                the 5x5 convolution
            F5 is the number of filters in the 5x5 convolution
            FPP is the number of filters in the 1x1 convolution after
            the max pooling
    All convolutions inside the inception block should use a
    rectified linear activation (ReLU)
    Returns: the concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters
    init = K.initializers.he_normal(seed=None)

    cnn1x1 = K.layers.Conv2D(filters=F1, kernel_size=1,
                             padding='same', activation='relu',
                             kernel_initializer=init)(A_prev)
    cnn3x3 = K.layers.Conv2D(filters=F3R, kernel_size=1,
                             padding='same', activation='relu',
                             kernel_initializer=init)(A_prev)
    cnn3x3 = K.layers.Conv2D(filters=F3, kernel_size=3,
                             padding='same', activation='relu',
                             kernel_initializer=init)(cnn3x3)
    cnn5x5 = K.layers.Conv2D(filters=F5R, kernel_size=1,
                             padding='same', activation='relu')(A_prev)
    cnn5x5 = K.layers.Conv2D(filters=F5, kernel_size=5,
                             padding='same', activation='relu',
                             kernel_initializer=init)(cnn5x5)
    pooling_ib = K.layers.MaxPool2D(pool_size=[3, 3], strides=1,
                                    padding='same')(A_prev)
    pooling_ib = K.layers.Conv2D(
            filters=FPP,
            kernel_size=1,
            padding='same',
            activation='relu',
            kernel_initializer=init)(pooling_ib)
    output = K.layers.concatenate([cnn1x1, cnn3x3, cnn5x5, pooling_ib])
    return output
