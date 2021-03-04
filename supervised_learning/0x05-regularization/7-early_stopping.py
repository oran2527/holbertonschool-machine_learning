#!/usr/bin/env python3
""" Early Stopping """


def early_stopping(cost, opt_cost, threshold, patience, count):
    """ Early Stopping """
    stop = False
    rest = opt_cost - cost
    if rest > threshold:
        count = 0
    else:
        count += 1
    if patience == count:
        stop = True
    return stop, count
