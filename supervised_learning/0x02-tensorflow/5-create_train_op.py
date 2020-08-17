#!/usr/bin/env python3
""" 5. Train_Op
function def create_train_op(loss, alpha):
that creates the training operation for the network
loss is the loss of the network’s prediction
alpha is the learning rate
Returns: an operation that trains the network using gradient descent
"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """ fn def create_train_op(loss, alpha):
    that creates the training operation for the network
    loss is the loss of the network’s prediction
    alpha is the learning rate
    Returns: an operation that trains the network using gradient descent
    """
    alpha = tf.train.GradientDescentOptimizer(alpha)
    loss = alpha.minimize(loss)
    return(loss)
