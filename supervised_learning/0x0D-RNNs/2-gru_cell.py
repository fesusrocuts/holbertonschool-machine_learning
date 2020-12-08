"""
class GRUCell that represents a gated recurrent unit:

    class constructor def __init__(self, i, h, o):
        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs
        Creates the public instance attributes Wz, Wr, Wh, Wy, bz, br, bh, by that represent the weights and biases of the cell
            Wzand bz are for the update gate
            Wrand br are for the reset gate
            Whand bh are for the intermediate hidden state
            Wyand by are for the output
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

class GRUCell:
    """class GRUCell"""

    def __init__(self, i, h, o):
        """fn __init__"""
        self.Wz = np.random.normal(size=(h + i, h))
        self.Wr = np.random.normal(size=(h + i, h))
        self.Wh = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """fn forward"""
        self.hidden_concat = np.concatenate((h_prev, x_t), axis=1)
        self.updateGate = self.sigmoid(np.matmul(self.hidden_concat, self.Wz) + self.bz)
        self.resetGate = self.sigmoid(np.matmul(self.hidden_concat, self.Wr) + self.br)
        self.gate_hidden_concat = np.concatenate((self.resetGate * h_prev, x_t), axis=1)
        self.h_prop = np.tanh(np.matmul(self.gate_hidden_concat, self.Wh) + self.bh)
        self.h_next = self.updateGate * self.h_prop * (1 - self.updateGate) * h_prev
        self.y = self.softmax(np.matmul(self.h_next, self.Wy) + self.by)
        return self.h_next, self.y

    def softmax(self, y):
        """fn softmax"""
        return np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)

    def sigmoid(self, x):
        """fn sigmoid"""
        return 1 / (1 + np.exp(-x))