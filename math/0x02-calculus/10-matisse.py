#!/usr/bin/env python3
"""
Write a function def poly_derivative(poly):
that calculates the derivative of a polynomial:
"""


def poly_derivative(poly):
    """
    Write a function def poly_derivative(poly):
    that calculates the derivative of a polynomial:
    """
    if type(poly) is list:
        if len(poly) is 0:
            return None
        else:
            return [0]
    else:
        return None
