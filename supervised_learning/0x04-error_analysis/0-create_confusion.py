#!/usr/bin/env python3
""" 0. Create Confusion
fn def create_confusion_matrix(labels, logits):
that creates a confusion matrix:
labels is a one-hot numpy.ndarray of shape (m, classes)
containing the correct labels for each data point
m is the number of data points
classes is the number of classes
logits is a one-hot numpy.ndarray of shape (m, classes)
containing the predicted labels
Returns: a confusion numpy.ndarray of shape (classes, classes)
with row indices representing the correct labels and column
indices representing the predicted labels
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """ 0. Create Confusion"""
    return np.matmul(labels.T, logits)
