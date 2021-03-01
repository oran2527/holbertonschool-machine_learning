#!/usr/bin/env python3
""" Sensitivity """
import numpy as np


def sensitivity(confusion):
    """ Sensitivy """
    conf = np.zeros(confusion.shape[0])
    for i in range(confusion.shape[0]):
        tp = confusion[i][i]
        fn = np.sum(confusion, axis=1) - tp
        conf[i] = tp / (tp + fn[i])
    return conf
