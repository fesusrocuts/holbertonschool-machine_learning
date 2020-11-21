#!/usr/bin/env python3
"""
6. Bayesian Optimization with GPyOpt mandatory

Write a python script that optimizes a machine learning model of your choice using GPyOpt:

    Your script should optimize at least 5 different hyperparameters. E.g. learning rate, number of units in a layer, dropout rate, L2 regularization weight, batch size
    Your model should be optimized on a single satisficing metric
    Your model should save a checkpoint of its best iteration during each training session
        The filename of the checkpoint should specify the values of the hyperparameters being tuned
    Your model should perform early stopping
    Bayesian optimization should run for a maximum of 30 iterations
    Once optimization has been performed, your script should plot the convergence
    Your script should save a report of the optimization to the file 'bayes_opt.txt'
    There are no restrictions on imports

Once you have finished your script, write a blog post describing your approach to this task. Your blog post should include:

    A description of what a Gaussian Process is
    A description of Bayesian Optimization
    The particular model that you chose to optimize
    The reasons you chose to focus on your specific hyperparameters
    The reason you chose your satisficing matric
    Your reasoning behind any other approach choices
    Any conclusions you made from performing this optimization
    Final thoughts

Your posts should have examples and at least one picture, at the top. Publish your blog post on Medium or LinkedIn, and share it at least on LinkedIn.

When done, please add all URLs below (blog post, tweet, etc.)

Please, remember that these blogs must be written in English to further your technical ability in a variety of settings.
"""
import numpy as np
from scipy.stats import norm}
from datetime import datetime
from scipy.optimize import minimize
from math import exp, fabs, sqrt, log, pi
GP = __import__('2-gp').GaussianProcess


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
