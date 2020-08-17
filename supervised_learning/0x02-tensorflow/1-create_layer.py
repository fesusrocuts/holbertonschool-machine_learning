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
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior() 


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
        initializer = tf.contrib.layers.variance_scaling_initializer(mode="fan_avg", distribution="normal")
        layer = (tf.layers.Dense(units=n, activation=activation,
                kernel_initializer=initializer, name="layer"))
        return layer(prev)
    except Exception as e:
        #tf.compat.v1.disable_v2_behavior()
        #tf.compat.v1.layers.Layer(
        #    trainable=True, name=None, dtype=None, **kwargs
        #)
        """
        initializer = tf.keras.initializers.VarianceScaling(mode="fan_avg", distribution="normal")
        layer = tf.compat.v1.layers.Dense(
            units=n, activation=activation, use_bias=True, kernel_initializer=initializer,
            bias_initializer=tf.zeros_initializer(), kernel_regularizer=None,
            bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
            bias_constraint=None, trainable=True, name="layer", **kwargs
        )
        """
        initializer = tf.keras.initializers.VarianceScaling(mode="fan_avg", distribution="normal")
        layer = (tf.compat.v1.layers.Dense(units=n, activation=activation,
                kernel_initializer=initializer, name="layer"))
        return layer(prev)
