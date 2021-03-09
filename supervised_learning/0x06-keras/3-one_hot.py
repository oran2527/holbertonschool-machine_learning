#!/usr/bin/env python3
""" One hod encode using keras utils """
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """ One hod encode using keras utils """
    matrix = K.utils.to_categorical(labels, classes)
    return matrix
