from collections import namedtuple

""" Experience object: data for single trajectory.
    Stored in ReplayBuffer for offline training.

    Fields
    ------
    traj: List of states
    r: reward; float.
    logr: log reward; tensor on device

    Generally, Experience is initialized with either:
    1. Minimal init: [traj, x, r, logr]. Minimum necessary for training.
    2. Full init, all. Used for replaybuffer / offline training.
"""

fields = [
    "traj",
    "x",
    "r",
    "logr",
]
Experience = namedtuple("Experience", fields, defaults=(None,) * len(fields))
