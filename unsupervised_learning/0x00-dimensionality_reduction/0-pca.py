#!/usr/bin/env python3
"""
fn pca that performs PCA on a dataset:
"""

import numpy as np


def pca(X, var=0.95):
    """
    placeholder
    """

    # list data for s and Vt variables
    U, s, Vt = np.linalg.svd(X)

    # cumulative for variance
    s_sum = np.sum(s)
    cumsum = np.cumsum(s)

    # limit for variance
    avg_var = cumsum / s_sum

    # find idx non-zero, grouped with element
    r = np.argwhere(avg_var >= var)[0, 0]

    W = Vt[:r + 1].T
    return (W)
