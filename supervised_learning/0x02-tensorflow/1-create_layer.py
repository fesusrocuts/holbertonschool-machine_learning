#!/usr/bin/env python3
""" 1. Layers
function def create_layer(prev, n, activation):
prev is the tensor output of the previous layer
n is the number of nodes in the layer to create
activation is the activation function that the layer should use
use tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
to implement He et. al initialization for the layer weights
each layer should be given the name layer
Returns: the tensor output of the layer
"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """ 1. Layers
    fn def create_layer,
    prev is the tensor output of the previous layer
    n is the number of nodes in the layer to create
    activation is the activation function that the layer should use
    use tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    to implement He et. al initialization for the layer weights
    each layer should be given the name layer
    Returns: the tensor output of the layer
    """

    try:
        initializer = \
            (tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
        return tf.layers.Dense(
            units=n,
            activation=activation,
            name='layer',
            kernel_initializer=initializer)(prev)
    except Exception as e:
        initializer = tf.compat.v1.keras.initializers.VarianceScaling(
            scale=1.0,
            mode='fan_avg',
            distribution='truncated_normal',
            seed=None
        )
        return tf.compat.v1.layers.Dense(
            units=n,
            activation=activation,
            use_bias=True,
            kernel_initializer=initializer,
            bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None, bias_constraint=None,
            trainable=True,
            name="layer")(prev)
