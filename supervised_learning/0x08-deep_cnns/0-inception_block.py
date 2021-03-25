#!/usr/bin/env python3
""" Inception block with Keras """
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """ Inception block with Keras """
    # Get number of filters for every layer
    F1, F3R, F3, F5R, F5, FPP = filters
    # Layer 1x1
    layer_0 = K.layers.Conv2D(F1, (1, 1), padding='same',
                              activation='relu')(A_prev)
    # Layer 1x1 then 3x3
    layer_1 = K.layers.Conv2D(F3R, (1, 1), padding='same',
                              activation='relu')(A_prev)
    layer_1 = K.layers.Conv2D(F3, (3, 3), padding='same',
                              activation='relu')(layer_1)
    # layer 1x1 then 5x5
    layer_2 = K.layers.Conv2D(F5R, (1, 1), padding='same',
                              activation='relu')(A_prev)
    layer_2 = K.layers.Conv2D(F5, (5, 5), padding='same',
                              activation='relu')(layer_2)
    # layer 3x3 then 1x1
    layer_3 = K.layers.MaxPooling2D((3, 3), strides=(1, 1),
                                    padding='same')(A_prev)
    layer_3 = K.layers.Conv2D(FPP, (1, 1), padding='same',
                              activation='relu')(layer_3)
    # Concatenate previous layers
    block = K.layers.concatenate([layer_0, layer_1, layer_2, layer_3],
                                 axis=3)
    return block
