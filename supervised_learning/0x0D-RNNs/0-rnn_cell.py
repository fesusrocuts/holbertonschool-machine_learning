"""
class RNNCell that represents a cell of a simple RNN:

    class constructor def __init__(self, i, h, o):
        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs
        Creates the public instance attributes Wh, Wy, bh, by that represent the weights and biases of the cell
            Wh and bh are for the concatenated hidden state and input data
            Wy and by are for the output
        The weights should be initialized using a random normal distribution in the order listed above
        The weights will be used on the right side for matrix multiplication
        The biases should be initialized as zeros
    public instance method def forward(self, h_prev, x_t): that performs forward propagation for one time step
        x_t is a numpy.ndarray of shape (m, i) that contains the data input for the cell
            m is the batche size for the data
        h_prev is a numpy.ndarray of shape (m, h) containing the previous hidden state
        The output of the cell should use a softmax activation function
        Returns: h_next, y
            h_next is the next hidden state
            y is the output of the cell
"""
import numpy as np

class RNNCell:
    """class RNNCell"""

    def __init__(self, i, h, o):
        """fn __init__"""
        self.i = i
        self.h = h
        self.o = o
        self.Wh = np.random.normal(size=(self.i + self.h, self.h))
        self.bh = np.zeros((1, self.h))
        self.Wy = np.random.normal(size=(self.h, self.o))
        self.by = np.zeros((1, self.o))

    def forward(self, h_prev, x_t):
        """fn forward"""
        self.hidden_concat = np.concatenate((h_prev.T, x_t.T), axis=0)
        self.h_next = np.tanh((np.matmul(self.hidden_concat.T, self.Wh)) + self.bh)
        self.y = self.softmax(np.matmul(self.h_next, self.Wy) + self.by)
        return self.h_next, self.y

    def softmax(self, y):
        """fn softmax"""
        return np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)