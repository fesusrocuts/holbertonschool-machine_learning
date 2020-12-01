#!/usr/bin/env python3
"""
3. Variational Autoencoder mandatory

Write a function def autoencoder(input_dims, hidden_layers, latent_dims): that creates a variational autoencoder:

    input_dims is an integer containing the dimensions of the model input
    hidden_layers is a list containing the number of nodes for each hidden layer in the encoder, respectively
        the hidden layers should be reversed for the decoder
    latent_dims is an integer containing the dimensions of the latent space representation
    Returns: encoder, decoder, auto
        encoder is the encoder model, which should output the latent representation, the mean, and the log variance, respectively
        decoder is the decoder model
        auto is the full autoencoder model
    The autoencoder model should be compiled using adam optimization and binary cross-entropy loss
    All layers should use a relu activation except for the mean and log variance layers in the encoder, which should use None, and the last layer in the decoder, which should use sigmoid
"""
import tensorflow.keras as keras
K = keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """fn that creates a variational autoencoder"""

    # This is our input image
    # input_dims
    # model, is the encoded representation of the input
    encoder_inputs = K.Input(shape=(input_dims,))
    # inputs
    for i in range(len(hidden_layers)):
        layer = K.layers.Dense(units=hidden_layers[i], activation='relu')
        if i == 0:
            z_mean = layer(encoder_inputs)
        else:
            z_mean = layer(z_mean)
    # rewrite layer
    layer = K.layers.Dense(units=latent_dims)
    z_mean = layer(z_mean)
    layer = K.layers.Dense(units=latent_dims)
    z_log_var = layer(z_mean)

    def sampling(args: tuple):
        """Reparameterization trick by sampling z from unit Gaussian"""
        # unpack
        z_mean, z_log_var = args
        # batch size
        batch_size = K.backend.shape(z_mean)[0]
        # latent dimension
        dim = K.backend.int_shape(z_mean)[1]
        # random normal vector with mean=0 and std=1.0
        epsilon = K.backend.random_normal(shape=batch_size, dim)
        return z_mean + K.backend.exp(0.5 * z_log_var) * epsilon

    z = K.layers.Lambda(sampling, output_shape=(latent_dims,))([z_mean, z_log_var])
    encoder = K.models.Model(encoder_inputs, [z, z_mean, z_log_var])

    # decoder model, is the lossy reconstruction of the input
    decoder_inputs = K.Input(shape=(latent_dims,))
    for i in range(len(hidden_layers) - 1, -1, -1):
        layer = K.layers.Dense(units=hidden_layers[i], activation='relu')
        if i == len(hidden_layers) - 1:
            z_mean = layer(decoder_inputs)
        else:
            z_mean = layer(z_mean)
    layer = K.layers.Dense(units=input_dims, activation='sigmoid')
    z_mean = layer(z_mean)
    decoder = K.models.Model(decoder_inputs, z_mean)

    # autoencoder
    z_mean = encoder(encoder_inputs)
    z_mean = decoder(z_mean)
    autoencoder = K.models.Model(encoder_inputs, z_mean)

    # Computes Kullback-Leibler divergence loss between y_true and y_pred.
    # https://www.tensorflow.org/api_docs/python/tf/keras/losses/KLDivergence
    # binary cross-entropy loss
    def KLDivergence(inputs, outputs):
        """fn KLDivergence"""
        loss = K.backend.binary_crossentropy(inputs, outputs)
        loss = K.backend.sum(loss, axis=1)
        KL_divergence = -0.5 * K.backend.sum(1 + z_log_var - K.backend.square(z_mean) - K.backend.exp(z_log_var), axis=-1)
        return loss + KL_divergence

    # compile
    autoencoder.compile(optimizer='Adam', loss=KLDivergence)

    return encoder, decoder, autoencoder
