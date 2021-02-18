#!/usr/bin/env python3
"""forward propagation"""

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    creates the forward propagation graph for the neural network:

    x is the placeholder for the input data
    layer_sizes is a list containing the number of nodes in each layer of
    the network
    activations is a list containing the activation functions for each
    layer of the network
    Returns: the prediction of the network in tensor form
    For this function, you should import your create_layer function with
    create_layer = __import__('1-create_layer').create_layer

    """

    if len(layer_sizes) and len(activations):
        if len(layer_sizes) == len(activations):
            A = x
            for i in range(len(layer_sizes)):
                A = create_layer(A, layer_sizes[i], activations[i])
            return A
    return None
