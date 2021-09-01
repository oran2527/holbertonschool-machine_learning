#!/usr/bin/env python3
""" From file """
import pandas as pd


def from_file(filename, delimiter):
    """ From file """
    df = pd.read_csv(filename, delimiter=delimiter)
    return df
