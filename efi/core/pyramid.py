"""Two-level field pyramid: coarse value sweeps extend the planning horizon.

With the log-domain readout fixing signal RANGE, the pyramid's job is
HORIZON: K sweeps at half resolution propagate value twice as far per tick
at a quarter of the cost. Per tick:

    1. pool beliefs/costs/rewards to the coarse grid (local 2x2 ops),
    2. run the same value recursion there,
    3. upsample coarse V (minus one coarse-step cost for intra-cell travel)
       as a LOWER BOUND initialization for the fine sweeps.

Pooling rules (each documented where used):
- blocked: coarse cell blocked iff ALL 4 fine cells blocked (optimistic --
  a half-open coarse cell is passable; the fine level corrects),
- cost q: mean over passable fine cells,
- reward R: max-pool (do not dilute a strong target).

All inter-level operations are local 2x2 pools / nearest-neighbor
upsampling; the coarse recursion is the same radius-1 stencil.
"""

from typing import Tuple

import numpy as np

from .desirability import VBIG, value_sweeps


def _pad_even(a: np.ndarray, fill) -> np.ndarray:
    H, W = a.shape
    if H % 2 == 0 and W % 2 == 0:
        return a
    out = np.full((H + H % 2, W + W % 2), fill, dtype=a.dtype)
    out[:H, :W] = a
    return out


def pool_blocked(blocked: np.ndarray) -> np.ndarray:
    """Coarse cell blocked iff all 4 fine cells blocked (optimistic)."""
    b = _pad_even(np.asarray(blocked, dtype=bool), True)
    return (b[0::2, 0::2] & b[1::2, 0::2] & b[0::2, 1::2] & b[1::2, 1::2])


def pool_cost(q: np.ndarray, blocked: np.ndarray) -> np.ndarray:
    """Mean cost over passable fine cells (blocked cells excluded)."""
    qf = _pad_even(np.asarray(q, dtype=np.float32), 0.0)
    bf = _pad_even(np.asarray(blocked, dtype=bool), True)
    passable = (~bf).astype(np.float32)
    s = (qf * passable)[0::2, 0::2] + (qf * passable)[1::2, 0::2] \
        + (qf * passable)[0::2, 1::2] + (qf * passable)[1::2, 1::2]
    n = passable[0::2, 0::2] + passable[1::2, 0::2] \
        + passable[0::2, 1::2] + passable[1::2, 1::2]
    return np.divide(s, n, out=np.zeros_like(s), where=n > 0).astype(np.float32)


def pool_reward(R_inj: np.ndarray) -> np.ndarray:
    """Max-pool rewards: a strong target must not be diluted."""
    r = _pad_even(np.asarray(R_inj, dtype=np.float32), -VBIG)
    return np.maximum.reduce([r[0::2, 0::2], r[1::2, 0::2],
                              r[0::2, 1::2], r[1::2, 1::2]]).astype(np.float32)


def upsample(coarse: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor 2x upsample, cropped to `shape`."""
    up = np.repeat(np.repeat(coarse, 2, axis=0), 2, axis=1)
    return up[:shape[0], :shape[1]].astype(np.float32)


def pyramid_value_sweeps(V, q, R_inj, walls_mask, lam, sweeps,
                         V_coarse=None, coarse_sweeps=None):
    """
    Two-level value computation. Runs `coarse_sweeps` (default: `sweeps`)
    at half resolution, uses the upsampled result as a lower-bound
    initialization (elementwise max with the warm-started fine V), then
    runs `sweeps` fine sweeps as usual.

    The coarse value is discounted by one coarse-step cost before use
    (intra-cell travel) and NEVER overrides walls or forbidden cells.

    Returns (V_fine, residuals_fine, V_coarse) -- persist V_coarse for
    warm starts.
    """
    walls = np.asarray(walls_mask) > 0
    cb = pool_blocked(walls)
    cq = pool_cost(q, walls)
    cR = pool_reward(R_inj)
    # One coarse step spans 2 fine cells: costs double per coarse hop.
    cq2 = 2.0 * cq

    if V_coarse is None or V_coarse.shape != cb.shape:
        V_coarse = np.zeros(cb.shape, dtype=np.float32)
    V_coarse, _ = value_sweeps(V_coarse, cq2, cR, cb,
                               lam=lam, sweeps=int(coarse_sweeps or sweeps))

    # Lower-bound init: coarse promise minus one coarse-step of cost,
    # never on impassable cells.
    guess = upsample(V_coarse, np.asarray(q).shape) - cq2.mean()
    V0 = np.maximum(np.asarray(V, dtype=np.float32), guess)
    V0[walls] = -VBIG

    V_fine, residuals = value_sweeps(V0, q, R_inj, walls_mask, lam=lam,
                                     sweeps=int(sweeps))
    return V_fine, residuals, V_coarse
