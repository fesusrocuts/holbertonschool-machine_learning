#!/usr/bin/env python3
"""
fn builds the DenseNet-121 architecture
"""

import tensorflow.keras as K


dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    growth_rate is the growth rate
    compression is the compression factor
    You can assume the input data will have shape (224, 224, 3)
    All convolutions should be preceded by Batch Normalization and a rectified linear activation (ReLU), respectively
    All weights should use he normal initialization
    You may use:
        dense_block = __import__('5-dense_block').dense_block
        transition_layer = __import__('6-transition_layer').transition_layer
    Returns: the keras model
    """
    input_Data = K.Input(shape=(224, 224, 3))
    init = K.initializers.he_normal(seed=None)
    batchNorm_1 = K.layers.BatchNormalization()(input_Data)
    activation_1 = K.layers.Activation('relu')(batchNorm_1)
    cnn_1 = K.layers.Conv2D(filters=64,
                           kernel_size=7,
                           strides=2,
                           padding='same',
                           kernel_initializer=init)(activation_1)
    pooling_1 = K.layers.MaxPooling2D(pool_size=3,
                                   strides=2,
                                   padding='same')(cnn_1)
    output_2, nbFilters_2 = dense_block(pooling_1, 64, growth_rate, 6)
    output_3, nbFilters_3 = transition_layer(output_2, nbFilters_2, compression)

    output_4, nbFilters_4 = dense_block(output_3, nbFilters_3, growth_rate, 12)
    output_5, nbFilters_5 = transition_layer(output_4, nbFilters_4, compression)

    output_6, nbFilters_6 = dense_block(output_5, nbFilters_5, growth_rate, 24)
    output_7, nbFilters_7 = transition_layer(output_6, nbFilters_6, compression)

    output_8, nbFilters_8 = dense_block(output_7, nbFilters_7, growth_rate, 16)

    avg_Pool = K.layers.AveragePooling2D(pool_size=7,
                                        strides=7,
                                        padding='same')(output_8)
    outputs = K.layers.Dense(units=1000,
                             activation='softmax',
                             kernel_initializer=init)(avg_Pool)
    return K.models.Model(input_Data, outputs)
