#!/usr/bin/env python3
""" 0. Initialize Poisson """


class Poisson:
    """ Poisson class"""

    def __init__(self, data=None, lambtha=1.):
        """ constructor"""
        # approximations
        self.pi = 3.1415926536
        self.e = 2.7182818285

        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")

            self.lambtha = float(sum(data) / len(data))
        else:

            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")

            self.lambtha = float(lambtha)

    def pmf(self, k):
        """ Calculates the value of the PMF for a given number of OK"""
        k = int(k)

        if k < 0:
            return 0

        # Multiply elements one by one, is factorial
        # factorial eq fc, summation eq sm,
        factorial = 1
        for x in range(1, k + 1):
            factorial *= x

        return ((self.e ** (- self.lambtha)) *
                (self.lambtha ** k)) / factorial

    def cdf(self, k):
        """ Calculates the value of the CDF for a given number of OK"""
        k = int(k)
        if k < 0:
            return 0

        sm = 0
        for i in range(k + 1):
            factorial = 1
            for x in range(1, i + 1):
                factorial *= x
            sm += (self.lambtha ** i) / factorial
        return (self.e ** (-self.lambtha)) * sm
