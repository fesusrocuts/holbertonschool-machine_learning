#!/usr/bin/env python3
""" 14. Batch Normalization Upgraded
function def create_batch_norm_layer(prev, n, activation):
that creates a batch normalization layer for a neural
network in tensorflow:
prev is the activated output of the previous layer
n is the number of nodes in the layer to be created
activation is the activation function that should
be used on the output of the layer
you should use the tf.layers.Dense layer as the base
layer with kernal initializer
tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
your layer should incorporate two trainable parameters, gamma and beta,
initialized as vectors of 1 and 0 respectively
you should use an epsilon of 1e-8
Returns: a tensor of the activated output for the layer
"""


import numpy as np
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """ 14. Batch Normalization Upgraded"""
    return(None)
