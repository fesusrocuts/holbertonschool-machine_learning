#!/usr/bin/env python3
""" 8. RMSProp Upgraded
function def create_RMSProp_op(loss, alpha, beta2, epsilon):
that creates the training operation for a neural network in tensorflow
using the RMSProp optimization algorithm:
loss is the loss of the network
alpha is the learning rate
beta2 is the RMSProp weight
epsilon is a small number to avoid division by zero
Returns: the RMSProp optimization operation
"""


import tensorflow as tf
import matplotlib.pyplot as plt


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """ 8. RMSProp Upgraded"""
    return(tf.train.RMSPropOptimizer(alpha, decay=beta2, \
                                      epsilon=epsilon).minimize(loss))
