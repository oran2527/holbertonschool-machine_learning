#!/usr/bin/env python3
""" Save and load weights """
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """ Save and load weights """
    network.save_weights('./' + filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """ Save and load weights """
    network.load_weights('./' + filename)
    return None
