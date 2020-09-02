#!/usr/bin/env python3
""" 11. Save and Load Configuration"""


import tensorflow.keras as K


def save_config(network, filename):
    """save keras"""
    with open(filename, 'w+') as file:
        file.write(network.to_json())


def load_config(filename):
    """load keras"""
    with open(filename, 'r') as file:
        return K.models.model_from_json(file.read())
