#!/usr/bin/env python3
""" f1 score """
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """ f1 score """
    recall = sensitivity(confusion)
    pre = precision(confusion)
    f1 = 2 * (recall * pre) / (recall + pre)
    return f1
