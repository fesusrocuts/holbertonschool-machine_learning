#!/usr/bin/env python3
"""function builds ResNet-50 architecture"""


import tensorflow.keras as K


identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    inputdata: shape (224, 224, 3)
    conv layer inside block follwed by batch norm layer
     along same channels
     and a ReLU
    weights use he_normal initialization
    Returns: keras model
    """

    init = K.initializers.he_normal(seed=None)
    input_Data = K.layers.Input(shape=(224, 224, 3))
    output_1 = K.layers.Conv2D(filters=64,
                               kernel_size=7,
                               strides=2,
                               padding='same',
                               kernel_initializer=init)(input_Data)
    batchNorm_1 = K.layers.BatchNormalization()(output_1)
    activation_1 = K.layers.Activation('relu')(batchNorm_1)
    pooling_1 = K.layers.MaxPooling2D(pool_size=3,
                                      strides=2,
                                      padding='same')(activation_1)

    f1 = [64, 64, 256]
    pb_2 = projection_block(pooling_1, f1, 1)
    output_3 = identity_block(pb_2, f1)
    output_4 = identity_block(output_3, f1)

    f2 = [128, 128, 512]
    pb_5 = projection_block(output_4, f2)
    output_6 = identity_block(pb_5, f2)
    output_7 = identity_block(output_6, f2)
    output_8 = identity_block(output_7, f2)

    f3 = [256, 256, 1024]
    pb_9 = projection_block(output_8, f3)
    output_10 = identity_block(pb_9, f3)
    output_11 = identity_block(output_10, f3)
    output_12 = identity_block(output_11, f3)
    output_13 = identity_block(output_12, f3)
    output_14 = identity_block(output_13, f3)

    f4 = [512, 512, 2048]
    pb_15 = projection_block(output_14, f4)
    output_16 = identity_block(pb_15, f4)
    output_17 = identity_block(output_16, f4)

    avg_Pool = K.layers.AveragePooling2D(pool_size=7,
                                         strides=None,
                                         padding='same')(output_17)
    outputs = K.layers.Dense(units=1000,
                             activation='softmax',
                             kernel_initializer=init)(avg_Pool)
    return K.models.Model(input_Data, outputs)
