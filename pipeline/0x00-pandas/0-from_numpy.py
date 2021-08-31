#!/usr/bin/env python3
""" from numpy """
import pandas as pd


def from_numpy(array):
    """ from numpy """
    abc = list("ABCDEFGHIJKLMNOPQRSTUVW")
    cols = array.shape[1]
    return pd.DataFrame(array, columns=abc[:cols])
