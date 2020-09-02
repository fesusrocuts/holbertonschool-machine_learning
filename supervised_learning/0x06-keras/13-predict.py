#!/usr/bin/env python3
""" 13. Predict"""


import tensorflow.keras as K


def predict(network, data, verbose=False):
    """ 13. Predict"""
    return network.predict(data, verbose=verbose)
