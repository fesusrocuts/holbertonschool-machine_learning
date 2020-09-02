#!/usr/bin/env python3
""" 3. One Hot"""


import tensorflow.keras as K


def one_hot(Y, classes=None):
    """ 3. One Hot"""
    if classes is None:
        classes = max(Y) + 1
    return K.utils.to_categorical(Y, classes)
