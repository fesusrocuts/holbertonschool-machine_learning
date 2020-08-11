#!/usr/bin/env python3
"""  function def one_hot_decode(one_hot): that converts a
one-hot matrix into a vector of labels:"""
import numpy as np


def one_hot_decode(one_hot):
    """  function def one_hot_decode(one_hot): that converts a
    one-hot matrix into a vector of labels:"""
    if type(one_hot) is not np.ndarray:
        return None
    if len(one_hot.shape) is not 2:
        return None
    if not(np.amin(one_hot) == 0 and np.amax(one_hot) == 1):
        return None
    r = one_hot.shape[0]
    column = one_hot.shape[1]
    list = np.ndarray(shape=(column), dtype=int)
    for i in range(r):
        for j in range(column):
            if (one_hot[i][j] == 1):
                list[j] = i
    return (list)
