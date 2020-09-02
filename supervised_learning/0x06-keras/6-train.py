#!/usr/bin/env python3
""" 6. Early Stopping"""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """ 6. Early Stopping"""
    callbacks = []
    if early_stopping and validation_data:
        callbacks.append(K.callbacks.EarlyStopping(patience=patience))
    return network.fit(data, labels, batch_size=batch_size, epochs=epochs,
                       shuffle=shuffle, verbose=verbose,
                       validation_data=validation_data,
                       callbacks=callbacks)
