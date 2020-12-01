#!/usr/bin/env python3
"""
2. Convolutional Autoencoder mandatory

Write a function def autoencoder(input_dims, filters, latent_dims): that creates a convolutional autoencoder:

    input_dims is a tuple of integers containing the dimensions of the model input
    filters is a list containing the number of filters for each convolutional layer in the encoder, respectively
        the filters should be reversed for the decoder
    latent_dims is a tuple of integers containing the dimensions of the latent space representation
    Each convolution in the encoder should use a kernel size of (3, 3) with same padding and relu activation, followed by max pooling of size (2, 2)
    Each convolution in the decoder, except for the last two, should use a filter size of (3, 3) with same padding and relu activation, followed by upsampling of size (2, 2)
        The second to last convolution should instead use valid padding
        The last convolution should have the same number of filters as the number of channels in input_dims with sigmoid activation and no upsampling
    Returns: encoder, decoder, auto
        encoder is the encoder model
        decoder is the decoder model
        auto is the full autoencoder model
    The autoencoder model should be compiled using adam optimization and binary cross-entropy loss
"""
import tensorflow.keras as keras
K = keras


def autoencoder(input_dims, filters, latent_dims):
    """fn that creates a convolutional autoencoder"""

    # This is our input image
    # input_dims
    # model, is the encoded representation of the input
    encoder_inputs = K.Input(shape=(input_dims,))
    # inputs
    for i in range(len(filters)):
        layer = K.layers.Conv2D(filters=filters[i], kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        if i == 0:
            z_mean = layer(encoder_inputs)
        else:
            z_mean = layer(z_mean)
    layer = K.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='same')
    z_mean = layer(z_mean)
    encoder = K.models.Model(encoder_inputs, z_mean)

    # decoder model, is the lossy reconstruction of the input
    decoder_inputs = K.Input(shape=(latent_dims,))
    for i in range(len(filters) - 1, -1, -1):
        layer = K.layers.Conv2D(filters=filters[i], kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        if i == len(filters) - 1:
            z_mean = layer(decoder_inputs)
        else:
            z_mean = layer(z_mean)
    layer = K.layers.Conv2D(filters=input_dims[-1], kernel_size=(3, 3), strides=(1, 1), padding='same', activation='sigmoid')
    z_mean = layer(z_mean)
    decoder = K.models.Model(decoder_inputs, z_mean)

    # autoencoder
    z_mean = encoder(encoder_inputs)
    z_mean = decoder(z_mean)
    autoencoder = K.models.Model(encoder_inputs, z_mean)

    # compile
    autoencoder.compile(optimizer='Adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder
