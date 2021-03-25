#!/usr/bin/env python3
""" Dense block with Keras """
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """ Dense block with Keras """
    init = K.initializers.he_normal()

    for i in range(layers):
        batch = K.layers.BatchNormalization()(X)
        act = K.layers.Activation(K.activations.relu)(batch)
        conv = K.layers.Conv2D(growth_rate * 4, (1, 1), padding='same',
                               strides=1,
                               kernel_initializer=init)(act)
        batch = K.layers.BatchNormalization()(conv)
        act = K.layers.Activation(K.activations.relu)(batch)
        conv = K.layers.Conv2D(growth_rate, (3, 3), padding='same',
                               strides=1,
                               kernel_initializer=init)(act)
        X = K.layers.concatenate([X, conv], axis=3)
        nb_filters += growth_rate
    return X, nb_filters
