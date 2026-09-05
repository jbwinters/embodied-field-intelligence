"""Scalar tabular reference for the same locally available effect model.

It uses identical sensing, evidence, outcome factors and work support, but
assembles observation groups and continuation values with a dictionary.
This is a numerical/model-based control, not an independent SOTA contender.
"""

import math

import numpy as np


def scalar_continuation(weights, keys, action_values, temperature):
    groups = {}
    for w, key, values in zip(weights, keys, action_values):
        key = tuple(key)
        if key not in groups:
            groups[key] = [0.0, [0.0] * 5]
        groups[key][0] += float(w)
        for action in range(5):
            groups[key][1][action] += float(w * values[action])
    result = {}
    for key, (mass, values) in groups.items():
        values = [v / max(mass, 1e-30) for v in values]
        peak = max(values)
        result[key] = peak + temperature * math.log(
            sum(math.exp((v - peak) / temperature) for v in values) / 5
        )
    return np.asarray([result[tuple(key)] for key in keys]), len(groups)
