#!/usr/bin/env python3
""" Transition layer with Keras """
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """ Transition layer with Keras """
    init = K.initializers.he_normal()
    batch = K.layers.BatchNormalization()(X)
    act = K.layers.Activation(K.activations.relu)(batch)
    conv = K.layers.Conv2D(int(nb_filters * compression), (1, 1),
                           padding='same',
                           strides=1,
                           kernel_initializer=init)(act)
    avg_pool = K.layers.AveragePooling2D(pool_size=(2, 2),
                                         strides=2)(conv)
    return avg_pool, int(nb_filters * compression)
