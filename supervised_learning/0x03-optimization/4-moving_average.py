#!/usr/bin/env python3
""" 4. Moving Average
function def moving_average(data, beta):
that calculates the weighted moving average of a data set:
data is the list of data to calculate the moving average of
beta is the weight used for the moving average
Your moving average calculation should use bias correction
Returns: a list containing the moving averages of data
"""

import numpy as np


def moving_average(data, beta):
    """ 4. Moving Average
    function def moving_average(data, beta):
    that calculates the weighted moving average of a data set:
    data is the list of data to calculate the moving average of
    beta is the weight used for the moving average
    Your moving average calculation should use bias correction
    Returns: a list containing the moving averages of data
    """
    movingav = []
    a = 0
    for i in range(0, len(data)):
        a = beta * a + (1 - beta) * data[i]
        movingav.append(a / ((beta ** (i + 1)) - 1))
    return movingav
