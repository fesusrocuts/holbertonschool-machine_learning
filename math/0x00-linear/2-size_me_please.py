#!/usr/bin/env python3
import numpy as np


def matrix_shape(x):
    x = np.asarray(list(x))
    x = x.astype(int, copy=False)
    x2 = []
    for i in x.shape:
        x2.append(i)
    x = x2
    del x2
    return x


def matrix_shape1(x, c=[]):
    try:
        c.append(len(x))
        if type(x) is list:
            for x2 in x:
                matrix_shape(x2)
    except Exception as e:
        pass
    return c
