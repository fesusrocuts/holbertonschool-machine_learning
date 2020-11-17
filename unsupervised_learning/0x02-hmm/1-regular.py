#!/usr/bin/env python3
"""
1. Regular Chains
function def regular(P): that determines the steady state probabilities of a regular markov chain:

    P is a is a square 2D numpy.ndarray of shape (n, n) representing the transition matrix
        P[i, j] is the probability of transitioning from state i to state j
        n is the number of states in the markov chain
    Returns: a numpy.ndarray of shape (1, n) containing the steady state probabilities, or None on failure
"""


import numpy as np


def regular(P):
    """
    1. Regular Chains
    function def regular(P): that determines the steady state probabilities of a regular markov chain:

        P is a is a square 2D numpy.ndarray of shape (n, n) representing the transition matrix
            P[i, j] is the probability of transitioning from state i to state j
            n is the number of states in the markov chain
        Returns: a numpy.ndarray of shape (1, n) containing the steady state probabilities, or None on failure
    """
    if ((type(P).__module__ != np.__name__ or 
        len(P.shape) != 2 or P.shape[0] < 1 or 
        P.shape[0] != P.shape[1] or 
        np.where(P < 0, 1, 0).any())):
        return None
    states = [P]
    square_2D = P
    while np.where(square_2D != 0, 0, 1).any():
        square_2D = np.dot(P, square_2D)
        if any(np.allclose(square_2D, i) for i in states):
            return None
        states.append(square_2D)
    while True:
        back = square_2D
        square_2D = np.dot(P, square_2D)
        if np.array_equal(square_2D, back):
            return square_2D[0:1]
