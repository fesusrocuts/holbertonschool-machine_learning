#!/usr/bin/env python3
""" 3. Specificity
function def specificity(confusion): that calculates the
specificity for each class in a confusion matrix:
onfusion is a confusion numpy.ndarray of shape (classes, classes)
where row indices represent the correct labels and column
indices represent the predicted labels
classes is the number of classes
Returns: a numpy.ndarray of shape (classes,) containing
the specificity of each class
"""
import numpy as np


def specificity(confusion):
    """ 3. Specificity"""
    SX = np.sum(confusion, axis=0)
    SY = np.sum(confusion, axis=1)
    ST = np.sum(confusion)
    TD = ST - SX - SY + np.diagonal(confusion)
    TD_DF = sum_all - SY
    return(TD / TD_DF)
