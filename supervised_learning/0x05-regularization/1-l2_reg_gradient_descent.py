#!/usr/bin/env python3
""" 1. Gradient Descent with L2 Regularization
unction def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
that updates the weights and biases of a neural network using gradient
descent with L2 regularization:
Y is a one-hot numpy.ndarray of shape (classes, m) that contains the
correct labels for the data
classes is the number of classes
m is the number of data points
weights is a dictionary of the weights and biases of the neural network
cache is a dictionary of the outputs of each layer of the neural network
alpha is the learning rate
lambtha is the L2 regularization parameter
L is the number of layers of the network
The neural network uses tanh activations on each layer except the last,
which uses a softmax activation
The weights and biases of the network should be updated in place
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """ 1. Gradient Descent with L2 Regularization"""
    record = {}
    matmul = {}
    matmulT = {}
    cY = {}
    m = Y.shape[1]
    wg = weights.copy()
    pi = str(L)
    cY['cY'+pi] = cache['A' + pi] - Y
    record['record'+pi] = np.sum(cY['cY'+pi], axis=1, keepdims=True)/m
    matmul['matmul'+pi] = np.matmul(cache['A'+str(L - 1)],
                              cY['cY'+pi].T) / m
    matmulT['matmulT'+pi] = matmul['matmul'+pi].T
    weights['W'+pi] = (1+(lambtha/m))*wg['W'+pi] - alpha*matmulT['matmulT'+pi]
    weights['b'+pi] = (1+(lambtha/m))*wg['b'+pi] - alpha*record['record'+pi]
    for i in range(L - 1, 0, -1):
        pm = str(i+1)
        pl = str(i-1)
        p = str(i)
        cY['cY'+p] = np.matmul(wg['W'+pm].T, cY['cY'+pm])
        record['record'+p] = np.sum(cY['cY'+p], axis=1, keepdims=True)/m
        matmul['matmul'+p] = np.matmul(cache['A'+pl],
                                 cY['cY'+p].T) / m
        matmulT['matmulT'+p] = matmul['matmul'+p].T
        weights['W'+p] = (1+(lambtha/m))*wg['W'+p] - alpha*matmulT['matmulT'+p]
        weights['b'+p] = (1+(lambtha/m))*wg['b'+p] - alpha*record['record'+p]
    return(weights)
