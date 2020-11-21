#!/usr/bin/env python3
"""
1. Gaussian Process Prediction mandatory

Based on 0-gp.py, update the class GaussianProcess:

    Public instance method def predict(self, X_s):
    that predicts the mean and standard deviation of points
    in a Gaussian process:
        X_s is a numpy.ndarray of shape (s, 1) containing all
        of the points whose mean and standard deviation
        should be calculated
            s is the number of sample points
        Returns: mu, sigma
            mu is a numpy.ndarray of shape (s,) containing
            the mean for each point in X_s, respectively
            sigma is a numpy.ndarray of shape (s,) containing
            the variance for each point in X_s, respectively
"""
import numpy as np


class GaussianProcess:
    """class GaussianProcess"""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """constructor"""

        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """
        fn calculates the covariance kernel matrix between two matrices
        """
        return None

    def predict(self, X_s):
        """
        fn Process Prediction
        """
        return None, None
