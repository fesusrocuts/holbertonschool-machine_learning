#!/usr/bin/env python3
""" 13. Batch Normalization
function def batch_norm(Z, gamma, beta, epsilon):
that normalizes an unactivated output of a neural network using
batch normalization:
Z is a numpy.ndarray of shape (m, n) that should be normalized
m is the number of data points
n is the number of features in Z
gamma is a numpy.ndarray of shape (1, n) containing the scales
used for batch normalization
beta is a numpy.ndarray of shape (1, n) containing the offsets
used for batch normalization
epsilon is a small number used to avoid division by zero
Returns: the normalized Z matrix
"""


import numpy as np
import tensorflow as tf


def batch_norm(Z, gamma, beta, epsilon):
    """ 13. Batch Normalization"""
    Z_norm = (Z - np.mean(Z)) / ((np.std(Z) + epsilon)**0.5)
    return(gamma * Z_norm + beta)
