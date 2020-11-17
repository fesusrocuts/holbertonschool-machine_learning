#!/usr/bin/env python3
"""
2. Absorbing Chains mandatory

function def absorbing(P): that determines if a markov chain is absorbing:

    P is a is a square 2D numpy.ndarray of shape (n, n) representing the standard transition matrix
        P[i, j] is the probability of transitioning from state i to state j
        n is the number of states in the markov chain
    Returns: True if it is absorbing, or False on failure
"""


import numpy as np


def absorbing(P):
    """
    2. Absorbing Chains mandatory

    function def absorbing(P): that determines if a markov chain is absorbing:

        P is a is a square 2D numpy.ndarray of shape (n, n) representing the standard transition matrix
            P[i, j] is the probability of transitioning from state i to state j
            n is the number of states in the markov chain
        Returns: True if it is absorbing, or False on failure
    """
    if ((type(P).__module__ != np.__name__ or 
        len(P.shape) != 2 or P.shape[0] < 1 or 
        P.shape[0] != P.shape[1] or 
        np.where(P < 0, 1, 0).any() or 
        not np.where(np.isclose(P.sum(axis=1), 1), 1, 0).any())):
        return None
    mufflers = np.zeros(P.shape[0])
    prev = mufflers.copy()
    for i in range(P.shape[0]):
        if P[i][i] == 1:
            mufflers[i] = 1
    while (not np.array_equal(mufflers, prev)
           and mufflers.sum() != P.shape[0]):
        prev = mufflers.copy()
        for _muffler in P[:, np.nonzero(mufflers)[0]].T:
            mufflers[np.nonzero(_muffler)] = 1
    if P.shape[0] == mufflers.sum():
        return True
    return False
