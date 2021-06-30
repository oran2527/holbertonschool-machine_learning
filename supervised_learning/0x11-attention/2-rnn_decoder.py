#!/usr/bin/env python3
""" RNN Decoder """
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """ RNN Decoder """
    def __init__(self, vocab, embedding, units, batch):
        """ RNN Decoder """
        super(RNNDecoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """ RNN Decoder """
        # Self attention
        attention = SelfAttention(self.units)
        context, weights = attention(s_prev, hidden_states)
        # Concat context with x
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context, 1), x], -1)
        # GRU
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))

        x = self.F(output)

        return x, state
