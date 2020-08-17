#!/usr/bin/env python3
""" 3. Accuracy
function def calculate_accuracy(y, y_pred):
that calculates the accuracy of a prediction
"""

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """ 3. Accuracy
    fn def calculate_accuracy(y, y_pred):
    that calculates the accuracy of a prediction
    y is a placeholder for the labels of the input data
    y_pred is a tensor containing the network’s predictions
    Returns: a tensor containing the decimal accuracy of the prediction
    """
    eq = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    resutl = tf.reduce_mean(tf.cast(eq, tf.float64))
    return(resutl)
