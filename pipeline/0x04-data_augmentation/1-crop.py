#!/usr/bin/env python3
""" crop """
import tensorflow as tf


def crop_image(image, size):
    """ crop """
    crop = tf.random_crop(image, size)

    return (crop)
