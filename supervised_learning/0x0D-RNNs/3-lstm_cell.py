"""
class LSTMCell that represents an LSTM unit:

    class constructor def __init__(self, i, h, o):
        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs
        Creates the public instance attributes Wf, Wu, Wc, Wo, Wy, bf, bu, bc, bo, by that represent the weights and biases of the cell
            Wfand bf are for the forget gate
            Wuand bu are for the update gate
            Wcand bc are for the intermediate cell state
            Woand bo are for the output gate
            Wyand by are for the outputs
        The weights should be initialized using a random normal distribution in the order listed above
        The weights will be used on the right side for matrix multiplication
        The biases should be initialized as zeros
    public instance method def forward(self, h_prev, c_prev, x_t): that performs forward propagation for one time step
        x_t is a numpy.ndarray of shape (m, i) that contains the data input for the cell
            m is the batche size for the data
        h_prev is a numpy.ndarray of shape (m, h) containing the previous hidden state
        c_prev is a numpy.ndarray of shape (m, h) containing the previous cell state
        The output of the cell should use a softmax activation function
        Returns: h_next, c_next, y
            h_next is the next hidden state
            c_next is the next cell state
            y is the output of the cell
"""
import numpy as np


class LSTMCell:
    """class LSTMCell"""

    def __init__(self, i, h, o):
        """fn __init__"""
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """fn forward"""
        self.hidden_concat = np.concatenate((h_prev, x_t), axis=1)
        self.forgetGate = self.sigmoid(np.matmul(self.hidden_concat, self.Wf) + self.bf)
        self.updateGate = self.sigmoid(np.matmul(self.hidden_concat, self.Wu) + self.bu)
        self.outputGate = self.sigmoid(np.matmul(self.hidden_concat, self.Wo) + self.bo)
        self.intermediateCellState = np.tanh(np.matmul(self.hidden_concat, self.Wc) + self.bc)
        self.c_next = self.forgetGate * c_prev + self.updateGate * self.intermediateCellState
        self.h_next = self.outputGate * np.tanh(self.c_next)
        self.y = self.softmax(np.matmul(self.h_next, self.Wy) + self.by)
        return self.h_next, self.c_next, self.y
    
    def softmax(self, y):
        """fn softmax"""
        return np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)

    def sigmoid(self, x):
        """fn sigmoid"""
        return 1 / (1 + np.exp(-x))