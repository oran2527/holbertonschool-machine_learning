#!/usr/bin/env python3
""" Training keras model using mini-batch gradient descent"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """ Training keras model using mini-batch gradient descent"""
    if validation_data:
        if early_stopping:
            call_back = [K.callbacks.EarlyStopping(monitor="val_loss",
                                                   patience=patience)]
        else:
            call_back = None
        history = network.fit(data, labels, batch_size=batch_size,
                              epochs=epochs, verbose=verbose,
                              shuffle=shuffle,
                              validation_data=validation_data,
                              callbacks=call_back)
    else:
        history = network.fit(data, labels, batch_size=batch_size,
                              epochs=epochs, verbose=verbose,
                              shuffle=shuffle)
    return history
