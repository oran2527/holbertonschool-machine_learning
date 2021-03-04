#!/usr/bin/env python3
""" Dropout layer """
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """ Dropout layer """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    regularizer = tf.layers.Dropout(keep_prob)
    layer = tf.layers.Dense(n, activation, name='layer',
                            kernel_initializer=init,
                            kernel_regularizer=regularizer)
    out = layer(prev)
    return out
