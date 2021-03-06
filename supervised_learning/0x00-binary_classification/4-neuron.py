#!/usr/bin/env python3
""" Neuron class: defines a single neuron performing binary classification
with pivate instances attributes
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

    def sigmoid(self, Z):
        """fn sigmoid"""
        sigm = 1 / (1 + np.exp(-Z))
        return sigm

    def forward_prop(self, X):
        """fn forward_prop"""
        recta = np.matmul(self.W, X) + self.b
        H = self.sigmoid(recta)
        self.__A = H
        return (self.__A)

    def cost(self, Y, A):
        """fn cost"""
        m = Y.shape[1]
        num_lreg = -1 * (Y * np.log(A) + (1 - Y) *
                         np.log(1.0000001 - A))
        cost = np.sum(num_lreg)/m
        return (cost)

    def log_reg(self, y_r, y_p):
        """fn log_reg"""
        num_lreg = -1 * (y_r * np.log(y_p) + (1 - y_r) *
                         np.log(1.0000001 - y_p))
        return (num_lreg)

    def evaluate(self, X, Y):
        """fn evaluate"""
        self.forward_prop(X)
        prediction = np.where(self.__A >= 0.5, 1, 0)
        return (prediction, self.cost(Y, self.__A))
