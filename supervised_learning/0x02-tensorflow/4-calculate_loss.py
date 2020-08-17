#!/usr/bin/env python3
""" 4. Loss
function def calculate_loss(y, y_pred):
that calculates the softmax cross-entropy loss of a prediction:
y is a placeholder for the labels of the input data
y_pred is a tensor containing the network’s predictions
Returns: a tensor containing the loss of the prediction
"""

import tensorflow as tf


def calculate_loss(y, y_pred):
    """ fn def calculate_loss(y, y_pred):
    that calculates the softmax cross-entropy loss of a prediction:
    y is a placeholder for the labels of the input data
    y_pred is a tensor containing the network’s predictions
    Returns: a tensor containing the loss of the prediction
    """
    return(tf.losses.softmax_cross_entropy(y, y_pred))
