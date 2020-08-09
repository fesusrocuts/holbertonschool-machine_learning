#!/usr/bin/env python3
""" class Neuron that defines a single neuron performing binary classification
"""

import numpy as np


class Neuron:
    """ class Neuron """
    def __init__(self, nx):
        """ Settings for class Neuron"""
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
