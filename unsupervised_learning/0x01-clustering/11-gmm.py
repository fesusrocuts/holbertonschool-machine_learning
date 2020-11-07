#!/usr/bin/env python3
""" 11. GMM"""


import sklearn.mixture


def gmm(X, k):
    """
    11. GMM
    Returns: pi, m, S, clss, bic
    """
    mixture = sklearn.mixture.GaussianMixture(k).fit(X)
    return (mixture.weights_,
            mixture.means_,
            mixture.covariances_,
            mixture.predict(X),
            mixture.bic(X))
