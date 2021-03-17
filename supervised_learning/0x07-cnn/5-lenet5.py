#!/usr/bin/env python3
""" LeNet-5 using keras """
import tensorflow.keras as K


def lenet5(X):
    """ LeNet-5 using keras """
    # C1 Convolutional Layer
    conv1 = K.layers.Conv2D(filters=6, kernel_size=(5, 5),
                            activation='relu', input_shape=(28, 28, 1),
                            kernel_initializer='he_normal',
                            padding='same')(X)

    # S2 Pooling Layer
    pool1 = K.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv1)

    # C3 Convolutional Layer
    conv2 = K.layers.Conv2D(filters=16, kernel_size=(5, 5),
                            kernel_initializer='he_normal',
                            activation='relu', padding='valid')(pool1)

    # S4 Pooling Layer
    pool2 = K.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    # Flatten the CNN output so that we can connect it with
    # fully connected layers
    flat = K.layers.Flatten()(pool2)

    # C5 Fully Connected Convolutional Layer
    full_l1 = K.layers.Dense(units=120, activation='relu',
                             kernel_initializer='he_normal',)(flat)

    # FC6 Fully Connected Layer
    full_l2 = K.layers.Dense(units=84, activation='relu',
                             kernel_initializer='he_normal',)(full_l1)

    # Output Layer with softmax activation
    output = K.layers.Dense(units=10, activation='softmax',
                            kernel_initializer='he_normal',)(full_l2)

    model = K.Model(X, output)

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=K.optimizers.Adam(), metrics=['accuracy'])

    return model
