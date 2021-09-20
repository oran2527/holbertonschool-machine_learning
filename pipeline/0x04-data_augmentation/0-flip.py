#!/usr/bin/env python3
""" flip """
import tensorflow as tf


def flip_image(image):
    """ flip """
    flip = tf.image.flip_left_right(image)

    return (flip)
