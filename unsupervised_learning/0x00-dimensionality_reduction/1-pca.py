#!/usr/bin/env python3
"""
Write a function def pca(X, ndim):
that performs PCA on a dataset:
"""
import numpy as np


def pca(X, ndim):
    """
    Write a function def pca(X, ndim):
    that performs PCA on a dataset:
    """
    # list data for s and Vt variables
    X_mean = X - np.mean(X, axis=0)
    u, s, Vt = np.linalg.svd(X_mean)
    W = Vt[:ndim].T
    M = np.matmul(X_mean, W) 
    return M
