#!/usr/bin/env python3
"""
FaceVerification Class
"""
import tensorflow as tf


class FaceVerification:
    """
    FaceVerification Class
    """
    def __init__(self, model, database, identities):
        """
        class constructor
        def __init__(self, model_path, database, identities):
        model_path is the path to where the face
        verification embedding model is stored
        you will need to use with
        tf.keras.utils.CustomObjectScope({'tf': tf}):
        to load the model
        database is a numpy.ndarray of shape (d, e) containing
        all the face embeddings in the database
        d is the number of images in the database
        e is the dimensionality of the embedding
        identities is a list of length d containing the
        identities corresponding to the embeddings in database
        Sets the public instance attributes model,
        database and identities
        """
        with tf.keras.utils.CustomObjectScope({'tf': tf}):
            self.model = tf.keras.models.load_model(model)
        self.database = database
        self.identities = identities

    def embedding(self, images):

        """
        public instance method def embedding(self, images)::
        that calculates the face embedding of images
        images is a numpy.ndarray of shape (i, n, n, 3)
        containing the aligned images
        i is the number of images
        n is the size of the aligned images
        Returns: a numpy.ndarray of shape (i, e) containing the
        embeddings where e is the dimensionality of the embeddings
        """
        embeddings = self.model.predict(images)

        return embeddings

    def verify(self, image, tau=0.5):
        """
        public instance method
        def verify(self, image, tau=0.5)::
        image is a numpy.ndarray of shape (n, n, 3)
        containing the aligned image of the face to be verified
        n is the shape of the aligned image
        tau is the maximum euclidean distance used
        for verification
        Returns: (identity, distance), or (None, None) on failure
        identity is a string containing the identity of
        the verified face
        distance is the euclidean distance between the verified
        face embedding and the identified database embedding
        """
        i = self.identities
        d = self.database
        distance = np.sum(np.square(tf.broadcast_to.embeddings(i - d)))
        if (distance > tau):
            return(i, distance)
        else:
            return (None, None)
