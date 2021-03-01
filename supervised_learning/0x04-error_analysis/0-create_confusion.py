#!/usr/bin/env python3
""" Confusion Matrix """
import numpy as np


def create_confusion_matrix(labels, logits):
    """ creates a confusion matrix """
    c_matrix = np.zeros((labels.shape[1], labels.shape[1]))
    for i in range(len(labels)):
        c = 0
        p = 0
        for j in range(len(labels[i])):
            if labels[i][j] == 1:
                c = j
            if logits[i][j] == 1:
                p = j
        c_matrix[c][p] += 1
    return c_matrix
