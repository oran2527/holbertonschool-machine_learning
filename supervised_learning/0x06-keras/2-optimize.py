#!/usr/bin/env python3
""" Optimizer using keras """
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """ Optimizer using keras """
    opt = K.optimizers.Adam(alpha, beta1, beta2)
    network.compile(optimizer=opt, loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return None
