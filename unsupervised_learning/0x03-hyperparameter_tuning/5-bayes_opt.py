#!/usr/bin/env python3
"""
5. Bayesian Optimization mandatory

Based on 4-bayes_opt.py, update the class BayesianOptimization:

    Public instance method def optimize(self, iterations=100): that optimizes the black-box function:
        iterations is the maximum number of iterations to perform
        If the next proposed point is one that has already been sampled, optimization should be stopped early
        Returns: X_opt, Y_opt
            X_opt is a numpy.ndarray of shape (1,) representing the optimal point
            Y_opt is a numpy.ndarray of shape (1,) representing the optimal function value
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

    def optimize(self, iterations=100):
        """fn optimizes the black-box function"""
        return None, None
