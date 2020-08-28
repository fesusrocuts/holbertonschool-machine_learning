#!/usr/bin/env python3
""" 4. Forward Propagation with Dropout
function def dropout_forward_prop(X, weights, L, keep_prob):
that conducts forward propagation using Dropout:
X is a numpy.ndarray of shape (nx, m) containing the
input data for the network
nx is the number of input features
m is the number of data points
weights is a dictionary of the weights and biases
of the neural network
L the number of layers in the network
keep_prob is the probability that a node will be kept
All layers except the last should use the tanh activation function
The last layer should use the softmax activation function
Returns: a dictionary containing the outputs of each layer and
the dropout mask used on each layer (see example for format)
"""
import tensorflow as tf
import numpy as np


def tanh(v):
    """ tanh activation function"""
    tanh = np.tanh(v)
    return tanh


def softmax(v):
    """ softmax activation function"""
    t = np.exp(v)
    sum = np.sum(t, axis=0, keepdims=True)
    softmax = t / sum
    return softmax


def dropout_forward_prop(X, weights, L, keep_prob):
    """ 4. Forward Propagation with Dropout"""
    dropout = {}
    for i in range(0, L):
        if i == 0:
            dropout['A0'] = X
        else:
            matmul_0 = np.matmul(weights['W' + str(i)],
                                dropout['A' + str(i-1)])
            matmul = matmul_0 + weights['b' + str(i)]
            dropout['D' + str(i)] = np.random.rand(matmul.shape[0],
                                                 matmul.shape[1])
            dropout['D' + str(i)] = dropout['D' + str(i)] < keep_prob
            dropout['D' + str(i)] = int(dropout['D' + str(i)] == 'true')
            matmul = np.multiply(matmul, dropout['D' + str(i)])
            matmul = matmul / keep_prob
            if i == L:
                H_tmp = softmax(matmul)
            else:
                H_tmp = tanh(matmul)
            dropout['A' + str(i)] = H_tmp
    return (dropout)
