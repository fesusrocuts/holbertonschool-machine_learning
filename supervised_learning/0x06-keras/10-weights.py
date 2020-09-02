#!/usr/bin/env python3
""" 10. Save and Load Weights"""


import tensorflow.keras as K


def save_weights(network, filename, format='h5'):
    """save keras"""
    network.save_weights(filename, format)


def load_weights(network, filename):
    """load keras"""
    return network.load_weights(filename)
