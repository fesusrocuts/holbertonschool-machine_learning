#!/usr/bin/env python3
"""  7. RMSProp
function def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
that updates a variable using the RMSProp optimization algorithm:
alpha is the learning rate
beta2 is the RMSProp weight
epsilon is a small number to avoid division by zero
var is a numpy.ndarray containing the variable to be updated
grad is a numpy.ndarray containing the gradient of var
s is the previous second moment of var
Returns: the updated variable and the new moment, respectively
"""


import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """ 7. RMSProp"""
    s = (beta2 * s) + ((1 - beta2) * grad * grad)
    var = var - ((alpha * grad) / (((new_s)**0.5) + epsilon))
    return (var, s)
