#!/usr/bin/env python3
""" Save and load from json format """
import tensorflow.keras as K


def save_config(network, filename):
    """ Save and load from json format """
    json_conf = network.to_json()
    with open(filename, "w") as json_file:
        json_file.write(json_conf)


def load_config(filename):
    """ Save and load from json format """
    with open(filename, "r") as json_file:
        json_conf = json_file.read()
    network = K.models.model_from_json(json_conf)
    return network
