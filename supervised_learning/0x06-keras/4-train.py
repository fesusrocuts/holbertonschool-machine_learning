#!/usr/bin/env python3
""" 4. Train"""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    """ 4. Train"""
    return network.fit(data, labels, batch_size=batch_size, epochs=epochs,
                       shuffle=shuffle, verbose=verbose)
