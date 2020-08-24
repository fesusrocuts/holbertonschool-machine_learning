#!/usr/bin/env python3
""" 2. Shuffle Data
function def shuffle_data(X, Y): that shuffles the data points in
two matrices the same way
"""

import numpy as np


def shuffle_data(X, Y):
    """ 2. Shuffle Data
    function def shuffle_data(X, Y): that shuffles the data points in
    two matrices the same way
    """
    assert len(X) == len(Y)
    perm = np.random.permutation(len(X))
    return (X[perm], Y[perm])
