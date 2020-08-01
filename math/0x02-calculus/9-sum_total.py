#!/usr/bin/env python3
"""Write a function def summation_i_squared(n):
that calculates sum_{i=1}^{n} i^2: """



def summation_i_squared2(value, limUpper, n):
    value += limLower**2
    limLower += 1
    if limLower == n:
        return value
    return summation_i_squared2(value, limUpper, n)




def summation_i_squared(n):
    """ Write a function def summation_i_squared(n):
    that calculates sum_{i=1}^{n} i^2: """
    try:
        if type(n) is int:
            """
            value = 0
            limLower = 1
            limUpper = n + 1
            exp = 2

            summation_i_squared(n+1)

            for i in range(n):
                if type(limUpper) is int and limUpper > 1:
                    value += limLower**exp
                    limLower += 1
                    # print("value = ", value)
            else:
                return value
            """
            print(n)
            summation_i_squared2(0, 1, n)
            return value
        else:
            return None
    except Exception as e:
        return None
