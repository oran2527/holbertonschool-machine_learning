#!/usr/bin/env python3
""" pca color augmentation """
import tensorflow as tf
import numpy as np


def pca_color(image, alphas):
    """ pca color augmentation """
    img = tf.keras.preprocessing.image.img_to_array(image)
    orig_img = img.astype(float).copy()

    img = img / 255.0

    reshape_img = img.reshape(-1, 3)

    centered_img = reshape_img - np.mean(reshape_img, axis=0)
    cov_img = np.cov(centered_img, rowvar=False)
    eig_vals, eig_vecs = np.linalg.eig(cov_img)

    sort_perm = eig_vals[::-1].argsort()
    eig_vals[::-1].sort()
    eig_vecs = eig_vecs[:, sort_perm]

    m1 = np.column_stack((eig_vecs))
    m2 = np.zeros((3, 1))
    m2[:, 0] = alphas * eig_vals[:]
    vect = np.matrix(m1) * np.matrix(m2)

    for i in range(3):
        orig_img[..., i] += vect[i]

    orig_img = np.clip(orig_img, 0.0, 255.0)
    orig_img = orig_img.astype(np.uint8)

    return (orig_img)
