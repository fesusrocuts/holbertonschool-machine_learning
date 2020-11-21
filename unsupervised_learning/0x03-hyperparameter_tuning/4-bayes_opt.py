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

    def acquisition(self):
        """fn  calculates the next best sample location"""
        return None, None
