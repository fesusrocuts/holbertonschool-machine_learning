#!/usr/bin/env python3
""" function def one_hot_encode,that converts a numeric
label vector into a one-hot matrix"""
import numpy as np


def one_hot_encode(Y, classes):
    """ function def one_hot_encode,that converts a numeric
    label vector into a one-hot matrix"""
    matrix = np.zeros((classes, len(Y)))
    if type(classes) is not int or 1 > classes:
        return (None)
    if len(Y) == 0:
        return (None)
    if type(Y) is not np.ndarray:
        return (None)
    if  classes <= np.amax(Y):
        return (None)
    if np.amin(Y) > 0:
        return (None)
    for i2 in range(len(Y)):
        i = Y[i2]
        matrix[i][i2] = 1
    return (matrix)
