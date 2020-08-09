#!/usr/bin/env python3
""" class DeepNeuralNetwork that defines a deep
neural network performing binary classification
"""

import numpy as np


class DeepNeuralNetwork:
    """ class DeepNeuralNetwork"""
    def __init__(self, nx, layers):
        """ Settings for class Deep Neural Networks"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")
        self.nx = nx
        self.layers = layers
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        ly = layers.copy()
        ly.insert(0, nx)
        for l in range(1, self.__L + 1):
            if type(ly[l-1]) is not int or ly[(l-1)] < 0:
                raise TypeError("layers must be a list of positive integers")
            tmp = np.random.randn(ly[l], ly[l-1]) * (np.sqrt(2/ly[l-1]))
            self.__weights['W'+str(l)] = tmp
            self.__weights['b'+str(l)] = np.zeros((ly[l], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights
