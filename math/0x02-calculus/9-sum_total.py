#!/usr/bin/env python3
"""Write a function def summation_i_squared(n):
that calculates sum_{i=1}^{n} i^2: """


def summation_i_squared(n):
    """ Write a function def summation_i_squared(n):
    that calculates sum_{i=1}^{n} i^2: """
    try:
        if type(n) is int:
            value = 0
            limLower = 1
            limUpper = n + 1
            exp = 2
            for i in range(n):
                if type(limUpper) is int and limUpper > 1:
                    value += limLower**exp
                    limLower += 1
                    # print("value = ", value)
            else:
                return value
        else:
            return None
    except Exception as e:
        return None
