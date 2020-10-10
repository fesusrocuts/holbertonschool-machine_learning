#!/usr/bin/env python3
"""
TrainModel class
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
from triplet_loss import TripletLoss


class TrainModel():
    """
    that trains a model for face verification using triplet loss:
    """
    def __init__(self, model_path, alpha):
        """
        Create the class constructor def __init__(self, model_path, alpha):
        model_path is the path to the base face verification embedding model
            loads the model using with
            tf.keras.utils.CustomObjectScope({'tf': tf}):
            saves this model as the public instance method base_model
        alpha is the alpha to use for the triplet loss calculation
        Creates a new model:
            inputs: [A, P, N]
                A is a numpy.ndarray of shape (m, n, n, 3)
                    containing the aligned anchor images
                P is a numpy.ndarray of shape (m, n, n, 3)
                    containing the aligned positive images
                N is a numpy.ndarray of shape (m, n, n, 3)
                    containing the aligned negative images
                m is the number of images
                n is the size of the aligned images
            outputs: the triplet losses of base_model
            compiles the model with:
                Adam optimization
                no additional losses
            save this model as the public instance attribute training_model
        you can use from triplet_loss import TripletLoss
        """
        # loads model and saves model
        with tf.keras.utils.CustomObjectScope({'tf': tf}):
            self.base_model = K.models.load_model(model_path)
        self.base_model.save(base_model)

        # inputs
        A_imgs = tf.placeholder(tf.float32, (None, 96, 96, 3))
        P_imgs = tf.placeholder(tf.float32, (None, 96, 96, 3))
        N_imgs = tf.placeholder(tf.float32, (None, 96, 96, 3))
        inputs = [A_imgs, P_imgs, N_imgs]

        # outputs encode
        output_images = self.base_model(inputs)

        # loss layer
        triplet_Loss = TripletLoss(alpha)
        outputs = triplet_Loss(output_images)

        # prepare training model
        training_model = K.models.Model(inputs, outputs)

        # compile using Adam
        training_model.compile(optimizer='Adam')

        # save training model
        training_model.save('training_model')

    def train(self, triplets, epochs=5, batch_size=32,
              validation_split=0.3, verbose=True):
        """
        Create the public instance method
        def train(self, triplets, epochs=5, batch_size=32,
        validation_split=0.3, verbose=True):
        that trains self.training_model:
        triplets is a list of numpy.ndarrayscontaining the
        inputs to self.training_model
        epochs is the number of epochs to train for
        batch_size is the batch size for training
        validation_split is the validation split for training
        verbose is a boolean that sets the verbosity mode
        Returns: the History output from the training
        """
        record = self.training_mode.fit(triplets,
                                        validation_split=validation_split,
                                        epochs=epochs,
                                        batch_size=batch_size,
                                        verbose=verbose)
        return record

    def save(self, save_path):
        """
        Create the public instance method def
        save(self, save_path): that saves the base embedding model:
        save_path is the path to save the model
        Returns: the saved model
        """
        self.base_model.save(save_path)

    @staticmethod
    def f1_score(y_true, y_pred):
        """
        static method def f1_score(y_true, y_pred):
        that calculates the F1 score of predictions
        y_true - a numpy.ndarray of shape (m,) containing the correct labels
            m is the number of examples
        y_pred- a numpy.ndarray of shape (m,) containing the predicted labels
        Returns: The f1 score
        static method def accuracy(y_true, y_pred):
        y_true - a numpy.ndarray of shape (m,) containing the correct labels
            m is the number of examples
        y_pred- a numpy.ndarray of shape (m,) containing the predicted labels
        Returns: the accuracy
        """
        return 1

    @staticmethod
    def accuracy(y_true, y_pred):
        """
        static method def accuracy(y_true, y_pred):
        y_true - a numpy.ndarray of shape (m,)
        containing the correct labels
        m is the number of examples
        y_pred- a numpy.ndarray of shape (m,)
        containing the predicted labels
        Returns: the accuracy
        """
        return K.metrics.BinaryAccuracy(y_true, y_pred)

    def best_tau(selfself, images, identities, thresholds):
        """
        public instance method
        def best_tau(self, images, identities, thresholds):
        that calculates the best tau to use for a maximal F1 score
        images - a numpy.ndarray of shape (m, n, n, 3) containing
        the aligned images for testing
        m is the number of images
        n is the size of the images
        identities - a list containing the identities of each image in images
        thresholds - a 1D numpy.ndarray of distance thresholds (tau) to test
        Returns: (tau, f1, acc)
        tau- the optimal threshold to maximize F1 score
        f1 - the maximal F1 score
        acc - the accuracy associated with the maximal F1 score
        """
        return 1
