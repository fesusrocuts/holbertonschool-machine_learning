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
    All convolutions should be preceded by Batch Normalization and a
        rectified linear activation (ReLU), respectively
    All weights should use he normal initialization
    You may use:
        dense_block = __import__('5-dense_block').dense_block
        transition_layer = __import__('6-transition_layer').transition_layer
    Returns: the keras model
    """
    input_Data = K.Input(shape=(224, 224, 3))
    init = K.initializers.he_normal(seed=None)
    r = K.layers.BatchNormalization()(input_Data)
    r = K.layers.Activation('relu')(r)
    r = K.layers.Conv2D(growth_rate * 2, 7, 2, padding='same',
                        kernel_initializer=init)(r)
    r = K.layers.MaxPool2D(2)(r)
    r, f = dense_block(r, growth_rate * 2, growth_rate, 6)
    r, f = transition_layer(r, f, compression)
    r, f = dense_block(r, f, growth_rate, 12)
    r, f = transition_layer(r, f, compression)
    r, f = dense_block(r, f, growth_rate, 24)
    r, f = transition_layer(r, f, compression)
    r, f = dense_block(r, f, growth_rate, 16)
    r = K.layers.AvgPool2D(7)(r)
    r = K.layers.Dense(1000, kernel_initializer=init,
                       activation='softmax')(r)
    return K.Model(input_Data, r)
