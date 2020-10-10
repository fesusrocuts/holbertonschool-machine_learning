#!/usr/bin/env python3
"""
Tripletloss Class
inherits from tensorflow.keras.layers.Layer:
"""
import tensorflow as tf
import tensorflow.keras as K


class Tripletloss:
    """
    inherits from tensorflow.keras.layers.Layer:
    """

    def __init__(self, alpha, **kwargs):
        """
        Create the class constructor
        def __init__(self, alpha, **kwargs):
        alpha is the alpha value used to calculate the triplet loss
        sets the public instance attribute alpha
        """
        self.alpha = alpha
        super(Tripletloss, self).__init__(**kwargs)
        
    def triplet_loss(self, inputs):
        """
        Create the public instance method
            def triplet_loss(self, inputs):
        inputs is a list containing the anchor, positive and negative
        output tensors from the last layer of the model, respectively
        Returns: a tensor containing the triplet loss values
        """
        A, P, N = inputs

        # anchor, positive and negative image
        subtracted_1 = K.layers.Subtract()([A, P])
        subtracted_2 = K.layers.Subtract()([A, N])

        sum_p = K.backend.sum(K.backend.square(subtracted_1), axis=1)
        sum_n = K.backend.sum(K.backend.square(subtracted_2), axis=1)

        subtracted_loss = K.layers.Subtract()([sum_p, sum_n]) + self.alpha
        
        return K.backend.maximum(subtracted_loss, 0)

    def call(self, inputs):
        """
        Create the public instance method def call(self, inputs):
        inputs is a list containing the anchor, positive, and negative
        output tensors from the last layer of the model, respectively
        adds the triplet loss to the graph
        Returns: the triplet loss tensor
        """
        loss = self.triplet_loss(inputs)
        
        # compile over Model
        self.add_loss(loss)
        
        return loss
