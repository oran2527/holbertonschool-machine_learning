#!/usr/bin/env python3
""" Training keras model using mini-batch gradient descent"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, verbose=True, shuffle=False):
    """ Training keras model using mini-batch gradient descent"""
    if validation_data:
        if early_stopping or learning_rate_decay:
            call_backs = []
            if early_stopping:
                call_backs.append(K.callbacks.EarlyStopping(monitor="val_loss",
                                                            patience=patience))
            if learning_rate_decay:
                def scheduler(epoch):
                    """ Scheduler function """
                    return alpha * 1/(1 + decay_rate * epoch)
                call_backs.append(
                            K.callbacks.LearningRateScheduler(scheduler,
                                                              verbose=True))
        else:
            call_backs = None
        history = network.fit(data, labels, batch_size=batch_size,
                              epochs=epochs, verbose=verbose,
                              shuffle=shuffle,
                              validation_data=validation_data,
                              callbacks=call_backs)
    else:
        history = network.fit(data, labels, batch_size=batch_size,
                              epochs=epochs, verbose=verbose,
                              shuffle=shuffle)
    return history
