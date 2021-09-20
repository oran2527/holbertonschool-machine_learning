#!/usr/bin/env python3
""" brightness """
import tensorflow as tf


def change_brightness(image, max_delta):
    """ brightness """
    brightness = tf.image.adjust_brightness(image, max_delta)

    return (brightness)
