#!/usr/bin/env python3
"""
fn build a projection block
"""
import tensorflow.keras as K


def conv_batchnorm_relu(x, filters, kernel_size, strides=1):
    # Conv-BatchNorm-ReLU block
    
    x = K.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding = 'same')(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Activation('relu')(x)
    # x = K.layers.ReLU()(x)
    
    return x


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
    # Projection block
    # tensor == A_prev
    # filters == filters
    # stride == s
    # left stream
    F11, F3, F12 = filters
    x = conv_batchnorm_relu(A_prev, filters=F11, kernel_size=1, strides=s)
    x = conv_batchnorm_relu(x, filters=F3, kernel_size=3, strides=1)
    x = K.layers.Conv2D(filters=4*F11, kernel_size=1, strides=1)(x)
    x = K.layers.BatchNormalization()(x)

    # right stream
    shortcut = K.layers.Conv2D(filters=F12, kernel_size=1, strides=s)(A_prev)
    shortcut = K.layers.BatchNormalization()(shortcut)

    x = K.layers.Add()([x,shortcut])    #skip connection
    x = K.layers.Activation('relu')(x)
    # x = K.layers.ReLU()(x)

    return x


def projection_block2(A_prev, filters, s=2):
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
