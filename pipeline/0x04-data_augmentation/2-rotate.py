#!/usr/bin/env python3
""" rotate """
import tensorflow as tf


def rotate_image(image):
    """ rotate """
    rotate = tf.image.rot90(image, k=1)

    return (rotate)
