#!/usr/bin/env python3
""" 11. Learning Rate Decay
function def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
that updates the learning rate using inverse time decay in numpy:
alpha is the original learning rate
decay_rate is the weight used to determine the rate at which alpha will decay
global_step is the number of passes of gradient descent that have elapsed
decay_step is the number of passes of gradient descent that should occur
before alpha is decayed further
the learning rate decay should occur in a stepwise fashion
Returns: the updated value for alpha
"""


import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """ 11. Learning Rate Decay"""
    update = alpha / (1 + decay_rate * int(global_step / decay_step))
    return (update)
