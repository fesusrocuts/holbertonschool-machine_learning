#!/usr/bin/env python3
""" Write a function def matrix_shape(matrix):
that calculates the shape of a matrix:

You can assume all elements in the same dimension are
of the same type/shape
The shape should be returned as a list of integers
"""


def matrix_shape(x):
    """ fn matrix_shape"""
    try:
        if not x:
            return None
        if type(x[0]) is not list:
            return [len(x)]
        return [len(x)] + matrix_shape(x[0])
    except TypeError:
        print("Error with the matrix")
        raise
