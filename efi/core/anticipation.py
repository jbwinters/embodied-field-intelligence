"""Finite-horizon control with time-indexed hazard forecasts.

One backward update is a radius-1 stencil over four moves and waiting.
For horizon h, future costs can influence at most h cells. The terminal
value comes from the existing spatial planner; this is an entropy-regularized
finite-horizon extension, not an additional converged global planner.
"""

import numpy as np

from .desirability import VBIG

MOTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0))


def shift(a, dy, dx, fill=0):
    """Move an array by one displacement without wraparound."""
    out = np.full_like(a, fill)
    H, W = a.shape
    out[max(0, dy) : min(H, H + dy), max(0, dx) : min(W, W + dx)] = a[
        max(0, -dy) : min(H, H - dy), max(0, -dx) : min(W, W - dx)
    ]
    return out


def arrival_values(
    terminal,
    costs,
    hazards,
    walls,
    lam,
    hazard_cost,
    edge_risks=None,
    targets=None,
    target_reward=1.0,
):
    """Return action values at every cell for the first move.

    hazards[t] is occupancy AFTER the t+1-th environment transition. Each
    action pays destination cost exactly once. Waiting pays cost too. Rewards
    are supplied by the terminal spatial value, so intermediate pickups are
    not repeatedly credited. The hazard process is assumed independent of
    agent actions. All alternatives share the same horizon and temperature.

    Optional targets[t] is the probability of a TERMINATING reward at the
    arrival cell. Its value replaces continuation on collection, preventing
    repeatedly collecting the same reward by waiting. These marginal fields
    approximate uncertain encounters; they do not condition object beliefs
    on every hypothetical failed interception. Edge exchanges do not collect.
    """
    if len(hazards) < 1 or lam <= 0 or hazard_cost < 0:
        raise ValueError("positive horizon/lambda and nonnegative hazard cost required")
    if targets is not None and (len(targets) != len(hazards) or target_reward < 0):
        raise ValueError("matching target horizon and nonnegative reward required")
    # Spatial V includes its own arrival cost. Remove that one boundary
    # charge because the final explicit transition pays it below.
    value = np.asarray(terminal, dtype=np.float32).copy() + costs
    for t in reversed(range(len(hazards))):
        hazard = hazards[t]
        destination = value - costs - hazard_cost * np.asarray(hazard)
        if targets is not None:
            hit = np.clip(np.asarray(targets[t]), 0, 1)
            destination = (1 - hit) * value + hit * target_reward - costs
            destination -= hazard_cost * np.asarray(hazard)
        scores = np.stack([shift(destination, -dy, -dx, -VBIG) for dy, dx in MOTIONS])
        if edge_risks is not None:
            # A hazard can collide by swapping cells with the agent even
            # if their destinations differ. Flux carries that probability.
            scores -= hazard_cost * edge_risks[t]
        valid = np.stack([shift(~walls, -dy, -dx, False) for dy, dx in MOTIONS])
        valid &= ~walls[None, :, :]
        scores = np.where(valid, scores, -VBIG)
        peak = scores.max(axis=0)
        with np.errstate(under="ignore"):
            mass = (np.exp((scores - peak) / lam) * valid).sum(axis=0)
        degree = valid.sum(axis=0)
        value = peak + lam * np.log(np.maximum(mass, 1e-30) / np.maximum(degree, 1))
        value[walls] = -VBIG
    return scores.astype(np.float32)


def action_probabilities(scores, lam):
    """Softmax with exact masking of unavailable actions, including waits."""
    scores = np.asarray(scores, dtype=np.float64)
    valid = scores > -float(VBIG) / 2
    if not valid.any():
        raise ValueError("no available action")
    logits = (scores - scores[valid].max()) / lam
    with np.errstate(under="ignore", over="ignore"):
        weights = np.where(valid, np.exp(np.minimum(logits, 0)), 0)
    return weights / weights.sum()
