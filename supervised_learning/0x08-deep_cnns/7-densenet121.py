#!/usr/bin/env python3
""" Dense Net 121 with Keras"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """ Dense Net 121 with Keras"""
    init = K.initializers.he_normal()
    # Input layer
    X = K.Input((224, 224, 3))
    batch = K.layers.BatchNormalization()(X)
    act = K.layers.Activation(K.activations.relu)(batch)
    # Conv_1
    conv = K.layers.Conv2D(64, (7, 7), padding='same',
                           strides=2,
                           kernel_initializer=init)(act)
    # Max pooling layer
    max_pool = K.layers.MaxPooling2D((3, 3), strides=2,
                                     padding='same')(conv)
    # Dense_block_1
    dense, nb_filters = dense_block(max_pool, 64, growth_rate, 6)
    # Trans_layer_1
    trans_layer, nb_filters = transition_layer(dense, nb_filters, compression)
    # Dense_block_2
    dense, nb_filters = dense_block(trans_layer, nb_filters, growth_rate, 12)
    # Trans_layer_1
    trans_layer, nb_filters = transition_layer(dense, nb_filters, compression)
    # Dense_block_3
    dense, nb_filters = dense_block(trans_layer, nb_filters, growth_rate, 24)
    # Trans_layer_1
    trans_layer, nb_filters = transition_layer(dense, nb_filters, compression)
    # Dense_block_4
    dense, nb_filters = dense_block(trans_layer, nb_filters, growth_rate, 16)
    # Global avarage
    avg_pool = K.layers.AveragePooling2D((7, 7))(dense)
    # Dense layer with softmax
    softmax = K.layers.Dense(1000, activation='softmax')(avg_pool)
    # Dense Model
    dense_net = K.models.Model(inputs=X, outputs=softmax)
    return dense_net
