#!/usr/bin/env python3
""" 9. Save and Load Model """


import tensorflow.keras as K


def save_model(network, filename):
    """save keras"""
    network.save(filename)


def load_model(filename):
    """load keras"""
    return K.models.load_model(filename)
