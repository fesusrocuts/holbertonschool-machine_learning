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

    def sigmoid(self, Z):
        """fn sigmoid"""
        sigm = 1 / (1 + np.exp(-Z))
        return sigm

    def forward_prop(self, X):
        """fn forward propoagation"""
        for i in range(self.__L + 1):
            if i == 0:
                self.__cache['A0'] = X
            else:
                A_tmp = (np.matmul(self.__weights['W' + str(i)],
                                   self.__cache['A' + str(i - 1)])
                         + self.__weights['b' + str(i)])
                H_tmp = self.sigmoid(A_tmp)
                self.__cache['A' + str(i)] = H_tmp
        return (self.__cache['A'+str(self.__L)], self.__cache)

    def cost(self, Y, A):
        """fn cost"""
        m = Y.shape[1]
        num_lreg = -1 * (Y * np.log(A) + (1 - Y) *
                         np.log(1.0000001 - A))
        cost = np.sum(num_lreg)/m
        return (cost)
