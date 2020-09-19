#!/usr/bin/env python3
"""
fn build a projection block
"""

import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """
    You can assume the input data will have shape (224, 224, 3)
    All convolutions inside and outside the blocks should
        be followed by batch normalization along the channels
        axis and a rectified
        linear activation (ReLU), respectively.
    All weights should use he normal initialization
    You may use:
        identity_block =
            __import__('2-identity_block').identity_block
        projection_block =
            __import__('3-projection_block').projection_block
    Returns: the keras model
    """

    F11, F3, F12 = filters

    init = K.initializers.he_normal(seed=None)

    output_1 = K.layers.Conv2D(filters=F11, kernel_size=1, strides=s,
                               padding='same', kernel_initializer=init)(A_prev)
    batchNorm_1 = K.layers.BatchNormalization()(output_1)
    activation_1 = K.layers.Activation('relu')(batchNorm_1)

    output_2 = K.layers.Conv2D(filters=F3, kernel_size=3, padding='same',
                               kernel_initializer=init)(activation_1)
    batchNorm_2 = K.layers.BatchNormalization()(output_2)
    activation_2 = K.layers.Activation('relu')(batchNorm_2)

    output_3 = K.layers.Conv2D(filters=F12, kernel_size=1,
                               padding='same',
                               kernel_initializer=init)(activation_2)
    batchNorm_3 = K.layers.BatchNormalization()(output_3)

    outside_short = K.layers.Conv2D(filters=F12, kernel_size=1,
                                    padding='same', strides=s,
                                    kernel_initializer=init)(A_prev)
    batchNorm_short = K.layers.BatchNormalization()(outside_short)

    addLayers = K.layers.Add()([batchNorm_3, batchNorm_short])
    return K.layers.Activation('relu')(addLayers)
