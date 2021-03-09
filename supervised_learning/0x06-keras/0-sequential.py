#!/usr/bin/env python3
""" NN with Keras library """
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ NN with Keras library """
    model = K.Sequential()
    for i in range(len(layers)):
        model.add(K.layers.Dense(layers[i],
                  activation=activations[i],
                  input_shape=(nx,),
                  kernel_regularizer=K.regularizers.l2(lambtha)))
        if i + 1 < len(layers):
            model.add(K.layers.Dropout(1 - keep_prob))
    return model
