"""
function def rnn(rnn_cell, X, h_0): that performs forward propagation for a simple RNN:

    rnn_cell is an instance of RNNCell that will be used for the forward propagation
    X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
        t is the maximum number of time steps
        m is the batch size
        i is the dimensionality of the data
    h_0 is the initial hidden state, given as a numpy.ndarray of shape (m, h)
        h is the dimensionality of the hidden state
    Returns: H, Y
        H is a numpy.ndarray containing all of the hidden states
        Y is a numpy.ndarray containing all of the outputs
"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """fn rnn"""
    t, m, i = X.shape
    m, h = h_0.shape
    H = np.zeros((t+1, m, h))
    Y = []
    H[0] = h_0
    for i in range(t):
        H[i+1], y_i = rnn_cell.forward(H[i], X[i])
        Y.append(y_i)
    return H, np.asarray(Y)