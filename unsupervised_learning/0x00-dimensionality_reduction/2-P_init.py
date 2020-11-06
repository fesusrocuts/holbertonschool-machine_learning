#!/usr/bin/env python3
"""
fn def P_init(X, perplexity):
that initializes all variables
required to calculate the P affinities in t-SNE:
"""
import numpy as np


def P_init(X, perplexity):
    """
    fn def P_init(X, perplexity):
    that initializes all variables
    required to calculate the P affinities in t-SNE
    """
    dtype='float64'
    (n, d) = X.shape
    X_sum = np.sum(np.square(X), axis=1)
    # Dot product of two arrays. Specifically, return ndarray
    dot = np.dot(X, X.T)
    D = np.add(np.add(-2 * dot, X_sum).T, X_sum)
    P = np.zeros([n, n], dtype=dtype)
    betas = np.ones([n, 1], dtype=dtype)
    # all Gaussian distributions should have
    H = np.log2(perplexity)

    return (D, P, betas, H)
