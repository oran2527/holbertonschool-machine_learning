#!/usr/bin/env python3
""" Self Attention """
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """ Self Attention """
    def __init__(self, units):
        """ Self Attention """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """ Self Attention """
        prev = self.W(tf.expand_dims(s_prev, 1))
        enc = self.U(hidden_states)
        e = self.V(tf.tanh(prev + enc))
        w = tf.nn.softmax(e, 1)
        context = w * hidden_states
        return tf.math.reduce_sum(context, 1), w
