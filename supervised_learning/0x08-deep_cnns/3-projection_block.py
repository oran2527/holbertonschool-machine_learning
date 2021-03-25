#!/usr/bin/env python3
""" Projection block with Keras """
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """ Projection block with Keras """
    F11, F3, F12 = filters
    init = K.initializers.he_normal()
    layer_0 = K.layers.Conv2D(F11, (1, 1), padding='same',
                              strides=s,
                              kernel_initializer=init)(A_prev)
    batch = K.layers.BatchNormalization()(layer_0)
    act = K.layers.Activation(K.activations.relu)(batch)
    layer_1 = K.layers.Conv2D(F3, (3, 3), padding='same',
                              kernel_initializer=init)(act)
    batch = K.layers.BatchNormalization()(layer_1)
    act = K.layers.Activation(K.activations.relu)(batch)
    layer_2 = K.layers.Conv2D(F12, (1, 1), padding='same',
                              kernel_initializer=init)(act)
    batch = K.layers.BatchNormalization()(layer_2)
    layer_3 = K.layers.Conv2D(F12, (1, 1), padding='same',
                              strides=s,
                              kernel_initializer=init)(A_prev)
    batch_short = K.layers.BatchNormalization()(layer_3)
    add = K.layers.Add()([batch, batch_short])
    act = K.layers.Activation(K.activations.relu)(add)

    return act
