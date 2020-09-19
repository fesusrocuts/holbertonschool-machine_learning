#!/usr/bin/env python3
"""
fn builds a transition layer
"""


import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    X is the output from the previous layer
    nb_filters is an integer representing the number of filters in X
    compression is the compression factor for the transition layer
    Your code should implement compression as used in DenseNet-C
    All weights should use he normal initialization
    All convolutions should be preceded by Batch Normalization and a rectified linear activation (ReLU), respectively
    Returns: The output of the transition layer and the number of filters within the output, respectively
    """

    init = K.initializers.he_normal(seed=None)
    compression_1 = int(compression * nb_filters)

    batchNorm_1 = K.layers.BatchNormalization()(X)
    activation_1 = K.layers.Activation('relu')(batchNorm_1)
    cnn1x1 = K.layers.Conv2D(filters=compression_1,
                              kernel_size=1,
                              padding='same',
                              kernel_initializer=init)(activation_1)
    avg_Pool = K.layers.AveragePooling2D(pool_size=2)(cnn1x1)
    return avg_Pool, compression_1
