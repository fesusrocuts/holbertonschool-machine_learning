#!/usr/bin/env python3
"""
Write a function def poly_integral(poly, C=0):
that calculates the integral of a polynomial:
"""


def poly_integral(poly, C=0):
    """
    Write a function def poly_integral(poly, C=0):
    that calculates the integral of a polynomial:
    """
    if type(poly) is list:
        if len(poly) is 0:
            return None
        else:
            return [0]
    else:
        return None
