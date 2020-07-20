#!/usr/bin/env python3


def matrix_transpose(x):
    """ fn matrix_transpose"""
    if type(x) is list:
        return [list(i) for i in zip(*x)]
    return []
