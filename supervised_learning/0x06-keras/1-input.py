#!/usr/bin/env python3
""" Neural network using Model class """
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ Neural network using Model class """
    inputs = K.Input(shape=(nx,))
    inp = inputs
    for i in range(len(layers)):
        if i + 1 < len(layers):
            layer = K.layers.Dense(layers[i],
                                   activation=activations[i],
                                   kernel_regularizer=K.regularizers.l2(
                                                        lambtha))(inp)
            dropout = (K.layers.Dropout(1 - keep_prob))(layer)
            inp = dropout
        else:
            layer = K.layers.Dense(layers[i],
                                   activation=activations[i],
                                   kernel_regularizer=K.regularizers.l2(
                                                        lambtha))(inp)
            output = layer
    model = K.Model(inputs=inputs, outputs=output)
    return model
