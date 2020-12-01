#!/usr/bin/env python3
"""
0. "Vanilla" Autoencoder mandatory

Write a function def autoencoder(input_dims, hidden_layers, latent_dims): that creates an autoencoder:

    input_dims is an integer containing the dimensions of the model input
    hidden_layers is a list containing the number of nodes for each hidden layer in the encoder, respectively
        the hidden layers should be reversed for the decoder
    latent_dims is an integer containing the dimensions of the latent space representation
    Returns: encoder, decoder, auto
        encoder is the encoder model
        decoder is the decoder model
        auto is the full autoencoder model
    The autoencoder model should be compiled using adam optimization and binary cross-entropy loss
    All layers should use a relu activation except for the last layer in the decoder, which should use sigmoid
"""
import tensorflow.keras as keras
K = keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """function that instantiates an autoencoder instance"""

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
    encoder = K.models.Model(encoder_inputs, z_mean)

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

    # compile
    autoencoder.compile(optimizer='Adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder
