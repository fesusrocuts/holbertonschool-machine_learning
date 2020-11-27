#!/usr/bin/env python3
"""
4. Bayesian Optimization - Acquisition mandatory

Based on 3-bayes_opt.py, update the class BayesianOptimization:

    Public instance method def acquisition(self):
    that calculates the next best sample location:
        Uses the Expected Improvement acquisition function
        Returns: X_next, EI
            X_next is a numpy.ndarray of shape (1,)
            representing the next best sample point
            EI is a numpy.ndarray of shape (ac_samples,)
            containing the expected improvement of each potential sample
    You may use from scipy.stats import norm
"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    class BayesianOptimization
    """

    def __init__(self, f, X_init, Y_init, bounds,
                 ac_samples, l=1, sigma_f=1, xsi=0.01, minimize=True):
        """constructor"""

        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1],
                               num=ac_samples)[..., np.newaxis]
        self.xsi = xsi
        self.minimize = minimize

    def acquisition2(self):
        """fn  calculates the next best sample location"""
        return None, None

    def acquisition(self):
        """
        fn  calculates the next best sample location
        determined by the maximum value of the acquisition function
        """

        mu, sigma = self.gp.predict(self.X_s)
        # is a numpy.ndarray of shape (1,)
        Sigma = np.zeros(sigma.shape)
        if self.minimize is True:
            Sigma_num = np.min(self.gp.Y) - mu - self.xsi
        else:
            Sigma_num = mu - np.max(self.gp.Y) - self.xsi

        for i in range(sigma.shape[0]):
            if sigma[i] > 0:
                Sigma[i] = Sigma_num[i] / sigma[i]
            else:
                Sigma[i] = 0

        # containing the expected improvement of each potential sample
        Expected = np.zeros(sigma.shape)
        for i in range(sigma.shape[0]):
            if sigma[i] > 0:
                Expected[i] = Sigma_num[i] * norm.cdf(Sigma_num[i]) + sigma[i] * norm.pdf(Sigma[i])
            else:
                Expected[i] = 0
        return self.X_s[np.argmax(Expected)], Expected