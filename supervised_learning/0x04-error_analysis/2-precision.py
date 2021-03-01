#!/usr/bin/env python3
""" Precision """
import numpy as np


def precision(confusion):
    """ Precision """
    prec = np.zeros(confusion.shape[0])
    for i in range(confusion.shape[0]):
        tp = confusion[i][i]
        fp = np.sum(confusion, axis=0) - tp
        prec[i] = tp / (tp + fp[i])
    return prec
