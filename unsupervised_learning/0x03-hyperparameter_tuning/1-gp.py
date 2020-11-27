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
        sqdist = np.sum(X1**2,1).reshape(-1,1) + \
                np.sum(X2**2,1) - 2*np.dot(X1, X2.T)
        return (self.sigma_f ** 2) * np.exp(-.5 * (1/(self.l ** 2.0)) * sqdist)

    def predict(self, X_s):
        """
        fn Process Prediction
        """
        K_p = self.kernel(self.X, X_s)
        K_pp = self.kernel(X_s, X_s)
        K_xx_inv = np.linalg.inv(self.K)
        mu = np.matmul(np.matmul(K_p.T, K_xx_inv), self.Y).reshape(-1)
        sigma_noise = K_pp - np.matmul(np.matmul(K_p.T, K_xx_inv), K_p)
        sigma = np.diag(sigma_noise)
        return mu, sigma
