#!/usr/bin/env python3
""" Write a function def matrix_shape(matrix):
that calculates the shape of a matrix:

You can assume all elements in the same dimension are
of the same type/shape
The shape should be returned as a list of integers
"""


def matrix_shape(x):
    """ fn matrix_shape"""
    a = len(x)
    b = len(x[0])
    try:
        d = len(x[0][0])
        c = [a, b, d]
    except:
        c = [a, b]
    return(c)
