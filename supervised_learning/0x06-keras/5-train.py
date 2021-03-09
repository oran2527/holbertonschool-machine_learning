#!/usr/bin/env python3
""" Training keras model using mini-batch gradient descent"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """ Training keras model using mini-batch gradient descent"""
    if validation_data:
        history = network.fit(data, labels, batch_size=batch_size,
                              epochs=epochs, verbose=verbose,
                              shuffle=shuffle,
                              validation_data=validation_data)
    else:
        history = network.fit(data, labels, batch_size=batch_size,
                              epochs=epochs, verbose=verbose,
                              shuffle=shuffle)
    return history
