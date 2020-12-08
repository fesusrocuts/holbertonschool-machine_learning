"""
function def deep_rnn(rnn_cells, X, h_0): that performs forward propagation for a deep RNN:

    rnn_cells is a list of RNNCell instances of length l that will be used for the forward propagation
        l is the number of layers
    X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
        t is the maximum number of time steps
        m is the batch size
        i is the dimensionality of the data
    h_0 is the initial hidden state, given as a numpy.ndarray of shape (l, m, h)
        h is the dimensionality of the hidden state
    Returns: H, Y
        H is a numpy.ndarray containing all of the hidden states
        Y is a numpy.ndarray containing all of the outputs
"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """fn deep_rnn"""
    t, m, i = X.shape
    l, m, h = h_0.shape
    k = len(rnn_cells)
    H = np.zeros((t + 1, k, m, h))
    Wy_0, Wy_1 = rnn_cells[k - 1].Wy.shape
    Y = np.zeros((t, m, Wy_1))
    
    for step in range(t):
        for layer in range(k):
            if step == 0:
                H[step, layer] = h_0[layer]
            if layer == 0:
                H[step + 1, layer], Y[step] = rnn_cells[layer].forward(H[step, layer], X[step])
            else:
                H[step + 1, layer], Y[step] = rnn_cells[layer].forward(H[step, layer], H[step + 1, layer - 1])

    return H, Y