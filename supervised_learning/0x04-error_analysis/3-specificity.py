#!/usr/bin/env python3
""" Specificity """
import numpy as np


def specificity(confusion):
    """ specificity """
    tp = np.diagonal(confusion)
    fp = np.sum(confusion, axis=0) - tp
    fn = np.sum(confusion, axis=1) - tp
    tn = np.sum(confusion) - (tp + fp + fn)
    spec = tn / (tn + fp)
    return spec
