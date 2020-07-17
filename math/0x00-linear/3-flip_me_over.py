#!/usr/bin/env python3
import numpy as np


def matrix_transpose(matrix):
    if type(matrix) is list:
        return np.array(matrix).transpose()
    return []
