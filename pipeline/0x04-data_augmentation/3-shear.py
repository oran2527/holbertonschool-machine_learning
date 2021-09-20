#!/usr/bin/env python3
""" shear """
import tensorflow as tf


def shear_image(image, intensity):
    """ shear """
    img = tf.keras.preprocessing.image.img_to_array(image)
    shear = tf.keras.preprocessing.image.random_shear(img, intensity,
                                                      row_axis=0,
                                                      col_axis=1,
                                                      channel_axis=2)
    shears = tf.keras.preprocessing.image.array_to_img(shear)

    return (shears)
