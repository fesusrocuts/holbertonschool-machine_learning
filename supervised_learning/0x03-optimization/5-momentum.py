#!/usr/bin/env python3
""" 5. Momentum
function def update_variables_momentum(alpha, beta1, var, grad, v):
that updates a variable using the gradient descent with
momentum optimization algorithm:
alpha is the learning rate
beta1 is the momentum weight
var is a numpy.ndarray containing the variable to be updated
grad is a numpy.ndarray containing the gradient of var
v is the previous first moment of var
Returns: the updated variable and the new moment, respectively
"""


import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """ 5. Momentum"""
    per = ((1 - beta1) * grad) + (beta1 * v)
    v = var - alpha * new_prom
    return (v, per)
