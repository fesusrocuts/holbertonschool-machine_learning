#!/usr/bin/env python3


def matrix_transpose(matrix):
    """ fn matrix_transpose"""
    if type(matrix) is list:
        return [list(i) for i in zip(*matrix)]
    return []
