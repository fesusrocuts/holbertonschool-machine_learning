#!/usr/bin/env python3
"""
fn builds dense block
"""


import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    X is the output from the previous layer
    nb_filters is an integer representing the number of filters in X
    growth_rate is the growth rate for the dense block
    layers is the number of layers in the dense block
    You should use the bottleneck layers used for DenseNet-B
    All weights should use he normal initialization
    All convolutions should be preceded by Batch Normalization and
        a rectified linear activation (ReLU), respectively
    Returns: The concatenated output of each layer within the
        Dense Block and the number of filters within the
        concatenated outputs, respectively
    """

    init = K.initializers.he_normal(seed=None)

    for blocks in range(layers):
        batchNorm_1 = K.layers.BatchNormalization()(X)
        activation_1 = K.layers.Activation('relu')(batchNorm_1)
        cnn_1x1 = K.layers.Conv2D(filters=growth_rate * 4,
                                  kernel_size=1,
                                  padding='same',
                                  kernel_initializer=init)(activation_1)
        batchNorm_2 = K.layers.BatchNormalization()(cnn_1x1)
        activation_2 = K.layers.Activation('relu')(batchNorm_2)
        # cnn 3x3
        next_X = K.layers.Conv2D(filters=growth_rate,
                                 kernel_size=3,
                                 padding='same',
                                 kernel_initializer=init)(activation_2)
        X = K.layers.concatenate([X, next_X])
        nb_filters += growth_rate
    return X, nb_filters
