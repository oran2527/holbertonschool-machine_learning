#!/usr/bin/env python3.6
""" Load env """
import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """ Load env """
    env = gym.make("FrozenLake-v0", desc=desc,
                   map_name=map_name, is_slippery=is_slippery)
    return env
