"""Log-odds Bayesian belief fields over target locations.

Replaces heuristic "scent" with a proper Bayes filter in log-odds form:

    L(v) = log P(target at v) / P(no target at v)

- Correction step (``logodds_correct``): local evidence from the observation
  window. Seeing a target adds ``l_pos``; seeing a cell EMPTY adds ``l_neg``
  (negative evidence — the piece scent fields were missing; without it the
  agent cannot disconfirm and keeps revisiting stale peaks).
- Prediction step (``logodds_predict``): diffusion in probability space
  models uncertainty growth while unobserved; relaxation toward a prior
  models "things may change when I am not looking".

All operations are local (radius-1 stencils) and run online, per tick.
"""

from typing import Iterable, Tuple

import numpy as np

from .diffusion import diffuse_masked


def sigmoid(L: np.ndarray) -> np.ndarray:
    """Log-odds -> probability, float32-stable."""
    return (1.0 / (1.0 + np.exp(-np.asarray(L, dtype=np.float32)))).astype(np.float32)


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Probability -> log-odds, clipped away from {0,1} for float32 stability."""
    p = np.clip(np.asarray(p, dtype=np.float32), eps, 1.0 - eps)
    return np.log(p / (1.0 - p)).astype(np.float32)


def logodds_correct(
    L: np.ndarray,
    pos_cells: Iterable[Tuple[int, int]],
    neg_cells: Iterable[Tuple[int, int]],
    l_pos: float = 8.0,
    l_neg: float = -8.0,
    l_min: float = -8.0,
    l_max: float = 8.0,
) -> np.ndarray:
    """
    Bayes correction from one observation window.

    Args:
        L: (H, W) log-odds field
        pos_cells: cells where the target WAS observed
        neg_cells: cells observed and found EMPTY (negative evidence)
        l_pos: evidence increment for a positive observation
        l_neg: evidence increment (negative) for an empty observation
        l_min, l_max: saturation clamp. REQUIRED: a belief pinned at
            +/-inf could never be disconfirmed by later evidence.

    Returns:
        Updated, clamped log-odds field (new array).
    """
    L = L.astype(np.float32, copy=True)
    for (y, x) in pos_cells:
        L[y, x] += l_pos
    for (y, x) in neg_cells:
        L[y, x] += l_neg
    return np.clip(L, l_min, l_max)


def logodds_predict(
    L: np.ndarray,
    walls_mask: np.ndarray,
    diff: float = 0.14,
    decay: float = 0.0,
    l_prior: float = -4.0,
    rho_prior: float = 0.002,
) -> np.ndarray:
    """
    Bayes prediction step: uncertainty growth while unobserved.

    Diffuses in probability space (masked by known walls: probability mass
    cannot leak through walls), then relaxes log-odds toward the prior.

    Args:
        L: (H, W) log-odds field
        walls_mask: boolean mask of known walls (True = wall)
        diff: spatial diffusion rate for probability mass
        decay: multiplicative probability decay per tick (usually 0)
        l_prior: log-odds every cell drifts toward when unobserved
        rho_prior: per-tick relaxation rate toward the prior

    Returns:
        Updated log-odds field (new array).
    """
    p = sigmoid(L)
    p = diffuse_masked(p, walls_mask, diff=diff, decay=decay, steps=1)
    L = logit(p)
    L = (1.0 - rho_prior) * L + rho_prior * np.float32(l_prior)
    return L.astype(np.float32)


def belief_to_expected_reward(L: np.ndarray, reward: float) -> np.ndarray:
    """
    Expected reward map: E[r | pick up at v] * P(target at v).

    Note: the controller composes raw probability maps weighted by LEARNED
    valences (the agent should not know env reward constants a priori);
    this helper is for callers that do know rewards (tests, analysis,
    value iteration over beliefs).
    """
    return (sigmoid(L) * np.float32(reward)).astype(np.float32)
