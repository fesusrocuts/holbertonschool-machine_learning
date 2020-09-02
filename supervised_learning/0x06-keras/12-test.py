#!/usr/bin/env python3
""" 12. Test """


import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """ 12. Test """
    return network.evaluate(data, labels, verbose=verbose)
