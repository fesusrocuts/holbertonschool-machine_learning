#!/usr/bin/env python3
"""
fn builds identity block
"""

import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    A_prev is the output from the previous layer
    filters is a tuple or list containing F11, F3, F12, respectively:
        F11 is the number of filters in the first 1x1 convolution
        F3 is the number of filters in the 3x3 convolution
        F12 is the number of filters in the second 1x1 convolution
    All convolutions inside the block should be followed
        by batch normalization along the channels axis and a
        rectified linear activation (ReLU), respectively.
    All weights should use he normal initialization
    Returns: the activated output of the identity block
    """
    F11, F3, F12 = filters
    init = K.initializers.he_normal(seed=None)

    output_1 = K.layers.Conv2D(filters=F11, kernel_size=1, padding='same',
                           kernel_initializer=init)(A_prev)
    batchNorm_1 = K.layers.BatchNormalization()(output_1)
    activation_1 = K.layers.Activation('relu')(batchNorm_1)

    output_2 = K.layers.Conv2D(filters=F3, kernel_size=3, padding='same',
                           kernel_initializer=init)(activation_1)
    batchNorm_2 = K.layers.BatchNormalization()(output_2)
    activation_2 = K.layers.Activation('relu')(batchNorm_2)

    output_3 = K.layers.Conv2D(filters=F12, kernel_size=1, padding='same',
                           kernel_initializer=init)(activation_2)
    batchNorm_3 = K.layers.BatchNormalization()(output_3)

    addLayers = K.layers.Add()([batchNorm_3, A_prev])
    return K.layers.Activation('relu')(addLayers)
