#!/usr/bin/env python3
"""Write a function def summation_i_squared(n):
that calculates sum_{i=1}^{n} i^2: """


def summation_i_squared2(value, limLower, n):
    value += limLower**2
    limLower += 1
    if limLower > n:
        return value
    return summation_i_squared2(value, limLower, n)


def summation_i_squared(n):
    """ Write a function def summation_i_squared(n):
    that calculates sum_{i=1}^{n} i^2: """
    try:
        if type(n) is int:
            return summation_i_squared2(0, 1, n)
        else:
            return None
    except Exception as e:
        return None
