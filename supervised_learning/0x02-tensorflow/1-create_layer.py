#!/usr/bin/env python3
"""create layer"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """
    prev is the tensor output of the previous layer
    n is the number of nodes in the layer to create
    activation is the activation function that the layer should use
    use tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    to implement He et. al initialization for the layer weights
    each layer should be given the name layer
    Returns: the tensor output of the layer

    """
    i = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    la = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=i,
        name='layer'
        )
    return la(prev)
