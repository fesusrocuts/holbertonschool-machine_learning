#!/usr/bin/env python3
""" 4. F1 score
function def f1_score(confusion): that calculates
the F1 score of a confusion matrix:
confusion is a confusion numpy.ndarray of shape (classes, classes)
where row indices represent the correct labels and column indices
represent the predicted labels
classes is the number of classes
Returns: a numpy.ndarray of shape (classes,) containing the
F1 score of each class
You may use sensitivity = __import__('1-sensitivity').sensitivity
and precision = __import__('2-precision').precision
"""
import numpy as np


def f1_score(confusion):
    """ 4. F1 score"""
    SX = np.sum(confusion, axis=0)
    SY = np.sum(confusion, axis=1)
    ST = np.sum(confusion)
    TP = np.diagonal(confusion)
    retrycall = TP / (SX)
    prec = TP / (SY)
    return(2 * (prec * retrycall) / (prec + retrycall))
