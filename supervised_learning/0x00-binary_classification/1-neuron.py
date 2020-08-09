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
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A
