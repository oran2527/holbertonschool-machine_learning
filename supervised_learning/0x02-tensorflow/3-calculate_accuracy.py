#!/usr/bin/env python3
""" Accuracy with TensorFlow """
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """ 
    y is a placeholder for the labels of the input data
    y_pred is a tensor containing the network’s predictions
    Returns: a tensor containing the decimal accuracy of the prediction

    """
    c_p = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    a = tf.reduce_mean(tf.cast(c_p, tf.float32))
    return a