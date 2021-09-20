#!/usr/bin/env python3
""" hue """
import tensorflow as tf


def change_hue(image, delta):
    """ hue """
    Hue = tf.image.adjust_hue(image, delta)

    return (Hue)
