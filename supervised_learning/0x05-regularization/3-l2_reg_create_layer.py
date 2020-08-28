#!/usr/bin/env python3
""" 3. Create a Layer with L2 Regularization
function def l2_reg_create_layer(prev, n, activation, lambtha):
that creates a tensorflow layer that includes L2 regularization:
prev is a tensor containing the output of the previous layer
n is the number of nodes the new layer should contain
activation is the activation function that should
be used on the layer
lambtha is the L2 regularization parameter
Returns: the output of the new layer
"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """ 3. Create a Layer with L2 Regularization"""
    vsi = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    l2r = tf.contrib.layers.l2_regularizer(lambtha)
    lDense = tf.layers.Dense(units=n, activation=activation,
                               kernel_initializer=vsi, name='layer',
                               kernel_regularizer=l2r)
    return(lDense(prev))
