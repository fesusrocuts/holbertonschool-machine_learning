#!/usr/bin/env python3
"""
6. Bayesian Optimization with GPyOpt mandatory
"""
import numpy as np


class bayesOpt:
    """
    6. Bayesian Optimization with GPyOpt mandatory
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
