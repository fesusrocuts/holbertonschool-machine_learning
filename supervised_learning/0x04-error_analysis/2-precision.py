#!/usr/bin/env python3
""" 2. Precision
function def precision(confusion): that calculates the
precision for each class in a confusion matrix:
confusion is a confusion numpy.ndarray of shape (classes, classes)
where row indices represent the correct labels and column indices
represent the predicted labels
classes is the number of classes
Returns: a numpy.ndarray of shape (classes,) containing the
precision of each class
"""
import numpy as np


def precision(confusion):
    """ 2. Precision"""
    return(np.diagonal(confusion) / np.sum(confusion, axis=0))
