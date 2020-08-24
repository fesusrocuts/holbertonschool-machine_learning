#!/usr/bin/env python3
""" 1. Function Normalize
function def normalize(X, m, s): that normalizes (standardizes) a matrix:
"""

import numpy as np


def normalize(X, m, s):
    """ 1. Function Normalize
    function def normalize(X, m, s): that normalizes (standardizes) a matrix:
    """
    return ((X - m) / (s))
