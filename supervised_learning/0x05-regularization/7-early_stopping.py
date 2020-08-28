#!/usr/bin/env python3
""" early_stopping"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """ early_stopping"""
    check = opt_cost - cost > threshold
    if not check:
        count += 1
    else:
        count = 0
    return (count >= patience, count)
