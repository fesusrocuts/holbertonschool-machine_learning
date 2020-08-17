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
        kwargs1 = {
            'units': n,
            'activation': activation,
            'name': 'layer', 'kernel_initializer': initializer
            }
        return tf.layers.Dense(kwargs1)(prev)
    except Exception as e:
        kwargs0 = {
            'unmodeits': "fan_avg",
            'distribution': "normal"
            }
        initializer = tf.keras.initializers.VarianceScaling(kwargs0)
        kwargs1 = {
            'units': n,
            'activation': activation,
            'name': 'layer', 'kernel_initializer': initializer
            }
        return tf.compat.v1.layers.Dense(kwargs1)(prev)
