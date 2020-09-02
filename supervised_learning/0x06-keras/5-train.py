#!/usr/bin/env python3
""" 5. Validate"""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """ 5. Validate"""
    return network.fit(data, labels, batch_size=batch_size, epochs=epochs,
                       shuffle=shuffle, verbose=verbose,
                       validation_data=validation_data)
