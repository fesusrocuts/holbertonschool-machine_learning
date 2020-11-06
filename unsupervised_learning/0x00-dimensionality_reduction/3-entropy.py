#!/usr/bin/env python3
"""
fn def HP that calculates the Shannon entropy and P
affinities relative to a data point:
"""
import numpy as np


def HP(Di, beta):
    """
    fn def HP that calculates the Shannon entropy and P
    affinities relative to a data point:
    """
    Pi = np.exp(beta * -Di)) / (np.sum(np.exp(beta * -Di))
    Hi = - np.sum(np.log2(Pi) * Pi)
    return (Hi, Pi)
