#!/usr/bin/env python3
"""
 2. Update Gaussian Process mandatory

Based on 1-gp.py, update the class GaussianProcess:

    Public instance method def update(self, X_new, Y_new):
    that updates a Gaussian Process:
        X_new is a numpy.ndarray of shape (1,)
        that represents the new sample point
        Y_new is a numpy.ndarray of shape (1,)
        that represents the new sample function value
        Updates the public instance attributes X, Y, and K
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

    def update(self, X_new, Y_new):
        """
        fn Update Gaussian Process mandatory
        """

        self.X = np.concatenate((self.X, X_new[..., np.newaxis]),
                                axis=0)
        self.Y = np.concatenate((self.Y, Y_new[..., np.newaxis]),
                                axis=0)
        self.K = self.kernel(self.X, self.X)
