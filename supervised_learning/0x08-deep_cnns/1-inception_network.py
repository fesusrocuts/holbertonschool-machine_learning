#!/usr/bin/env python3
"""
fn that builds the inception network
"""

import tensorflow.keras as K


inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    You can assume the input data will have shape (224, 224, 3)
    All convolutions inside and outside the inception block should
        use a rectified linear activation (ReLU)
    You may use inception_block =
        __import__('0-inception_block').inception_block
    Returns: the keras model
    """
    init = K.initializers.he_normal(seed=None)
    input_Data = K.Input(shape=(224, 224, 3))

    net = K.layers.Conv2D(filters=64, kernel_size=7, strides=2,
                          padding='same', activation='relu',
                          kernel_initializer=init)(input_Data)
    net = K.layers.MaxPool2D(pool_size=3, strides=2,
                             padding='same')(net)

    net = K.layers.Conv2D(filters=64, kernel_size=1,
                          padding='same', activation='relu',
                          kernel_initializer=init)(net)
    net = K.layers.Conv2D(filters=192, kernel_size=3,
                          padding='same', activation='relu',
                          kernel_initializer=init)(net)
    net = K.layers.MaxPool2D(pool_size=3, strides=2,
                             padding='same')(net)

    net = inception_block(net, [64, 96, 128, 16, 32, 32])
    net = inception_block(net, [128, 128, 192, 32, 96, 64])

    net = K.layers.MaxPool2D(pool_size=3, strides=2,
                             padding='same')(net)

    net = inception_block(net, [192, 96, 208, 16, 48, 64])
    net = inception_block(net, [160, 112, 224, 24, 64, 64])
    net = inception_block(net, [128, 128, 256, 24, 64, 64])
    net = inception_block(net, [112, 144, 288, 32, 64, 64])
    net = inception_block(net, [256, 160, 320, 32, 128, 128])

    net = K.layers.MaxPool2D(pool_size=3, strides=2,
                             padding='same')(net)

    net = inception_block(net, [256, 160, 320, 32, 128, 128])
    net = inception_block(net, [384, 192, 384, 48, 128, 128])

    avg_Pool = K.layers.AveragePooling2D(pool_size=7, strides=1)(net)

    drop_Out = K.layers.Dropout(.4)(avg_Pool)
    out_Data = K.layers.Dense(1000, activation='softmax',
                              kernel_initializer=init)(drop_Out)
    return K.Model(input_Data, out_Data)
