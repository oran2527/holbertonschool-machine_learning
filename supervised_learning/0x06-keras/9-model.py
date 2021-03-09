#!/usr/bin/env python3
""" Save and load models """
import tensorflow.keras as K


def save_model(network, filename):
    """ Save and load models """
    network.save(filename)


def load_model(filename):
    """ Save and load models """
    model = K.models.load_model('./' + filename)
    return model
