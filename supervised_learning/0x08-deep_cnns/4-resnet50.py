#!/usr/bin/env python3
""" ResNet-50 With Keras """
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """ ResNet-50 With Keras """
    X = K.Input((224, 224, 3))
    init = K.initializers.he_normal()
    conv = K.layers.Conv2D(64, (7, 7), padding='same',
                           strides=2,
                           kernel_initializer=init)(X)
    batch = K.layers.BatchNormalization()(conv)
    act = K.layers.Activation(K.activations.relu)(batch)
    max_pool = K.layers.MaxPooling2D((3, 3), strides=2,
                                     padding='same')(act)

    # Conv_2
    block = projection_block(max_pool, [64, 64, 256], 1)
    block = identity_block(block, [64, 64, 256])
    block = identity_block(block, [64, 64, 256])
    # Conv_3
    block = projection_block(block, [128, 128, 512], 2)
    block = identity_block(block, [128, 128, 512])
    block = identity_block(block, [128, 128, 512])
    block = identity_block(block, [128, 128, 512])
    # Conv_4
    block = projection_block(block, [256, 256, 1024], 2)
    block = identity_block(block, [256, 256, 1024])
    block = identity_block(block, [256, 256, 1024])
    block = identity_block(block, [256, 256, 1024])
    block = identity_block(block, [256, 256, 1024])
    block = identity_block(block, [256, 256, 1024])
    # Conv_5
    block = projection_block(block, [512, 512, 2048], 2)
    block = identity_block(block, [512, 512, 2048])
    block = identity_block(block, [512, 512, 2048])
    avg_pool = K.layers.AveragePooling2D(pool_size=(7, 7),
                                         strides=1)(block)
    softmax = K.layers.Dense(1000, activation='softmax')(avg_pool)
    res_net = K.models.Model(inputs=X, outputs=softmax)
    return res_net
