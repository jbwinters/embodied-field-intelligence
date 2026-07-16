"""Desirability-field value computation (linearly-solvable MDP form).

Value iteration as a LOCAL field operation. In Todorov's linearly-solvable
MDPs the desirability z(v) = exp(V(v)/lam) satisfies a linear, local
fixed-point equation:

    z(v) = exp(-q(v)/lam) * neighborAvg(z)(v)

i.e. optimal value iteration is literally masked diffusion with a pointwise
multiplicative cost term. We work ONLY in V-space (V = lam*log z) so that:

- values attenuate LINEARLY with distance (z attenuates exponentially and
  underflows on large grids) -- this is what removes the scaling ceiling;
- repulsors enter as additive state costs q(v), the mathematically correct
  way for penalties to shape paths (subtracting a diffused repulsor from an
  attractor field double-counts geometry);
- lam is a single risk/temperature dial: lam -> 0 recovers max-plus /
  worst-case / shortest-path planning; large lam is diffusive and
  exploratory. The optimal LMDP policy is a softmax over neighbor V/lam.

One sweep (all radius-1 stencil operations):

    V <- max(lse4(V), R_inj) - q

i.e. neighbor average of z in log space, reward injection, THEN state
costs. The ordering matters: injecting reward after subtracting q would
let a cell's reward erase its own cost (an aversive cell carrying an
epistemic bonus would become a free stepping stone). Arriving at a cell to
collect its reward still pays that cell's cost -- the first-exit value is
net of the running cost of the exit state.
"""

from typing import List, Optional, Tuple

import numpy as np

# Wall / invalid sentinel. Finite (not -inf) so the log-sum-exp never sees NaN.
VBIG = np.float32(1.0e9)
_TINY = np.float32(1.0e-30)


def lse_neighbor_avg(V: np.ndarray, passable: np.ndarray, lam: float) -> np.ndarray:
    """
    Log-domain neighbor average: lam * log( (1/n) * sum_u exp(V(u)/lam) )
    over the up-to-4 passable neighbors u of each cell.

    Stable: subtracts the per-cell max before exponentiating, so exponents
    are always <= 0. Cells with no passable neighbor get -VBIG.
    """
    V = np.asarray(V, dtype=np.float32)
    P = np.asarray(passable, dtype=np.float32)
    H, W = V.shape
    lam = np.float32(max(lam, 1e-6))

    vals = np.full((4, H, W), -VBIG, dtype=np.float32)
    valid = np.zeros((4, H, W), dtype=np.float32)
    # Same 4-shift stencil as diffuse_masked.
    vals[0, 1:, :] = V[:-1, :]
    valid[0, 1:, :] = P[:-1, :]
    vals[1, :-1, :] = V[1:, :]
    valid[1, :-1, :] = P[1:, :]
    vals[2, :, 1:] = V[:, :-1]
    valid[2, :, 1:] = P[:, :-1]
    vals[3, :, :-1] = V[:, 1:]
    valid[3, :, :-1] = P[:, 1:]

    vals = np.where(valid > 0, vals, -VBIG)
    m = vals.max(axis=0)
    with np.errstate(under="ignore"):
        e = np.exp((vals - m[None, :, :]) / lam) * valid
    s = e.sum(axis=0)
    n = valid.sum(axis=0)
    out = m + lam * (np.log(s + _TINY) - np.log(n + _TINY))
    return np.where(n > 0, out, -VBIG).astype(np.float32)


def value_sweeps(
    V: np.ndarray,
    q: np.ndarray,
    R_inj: np.ndarray,
    walls_mask: np.ndarray,
    lam: float,
    sweeps: int,
) -> Tuple[np.ndarray, List[float]]:
    """
    Run `sweeps` local value-iteration sweeps; return (V, residuals).

    Args:
        V: (H, W) value field to start from (warm start; zeros for cold)
        q: (H, W) nonnegative state costs, units of reward-per-step
           (trail, hazard, membrane, pain, aversive targets, step effort)
        R_inj: (H, W) reward injection: expected reward for terminating at
           each cell (-VBIG where no reward). Applied as V = max(V, R_inj).
        walls_mask: boolean, True = impassable
        lam: risk/temperature parameter (> 0); lam -> 0 is max-plus
        sweeps: number of relaxation sweeps this call

    Returns:
        (V, residuals): updated field and per-sweep max|dV| on passable
        cells -- the fixed-point tracking diagnostic.

    Convergence note: a passable region with NO reward injection anywhere
    (R_inj = -VBIG throughout an enclosed pocket) has fixed point -inf --
    V drains by q per sweep indefinitely and the residual never vanishes.
    Callers wanting a finite fixed point must provide a floor; the
    controller's belief-prior optimism floor (every cell might hold a
    target at prior rate) serves exactly this role.
    """
    Wm = np.asarray(walls_mask) > 0
    passable = ~Wm
    V = np.asarray(V, dtype=np.float32).copy()
    V[Wm] = -VBIG
    q = np.asarray(q, dtype=np.float32)
    R_inj = np.asarray(R_inj, dtype=np.float32)

    residuals: List[float] = []
    for _ in range(max(0, int(sweeps))):
        V_new = lse_neighbor_avg(V, passable, lam)
        V_new = np.maximum(V_new, R_inj)
        V_new = V_new - q
        V_new[Wm] = -VBIG
        # Residual over passable cells only; clip sentinels out.
        dv = np.abs(V_new[passable] - V[passable])
        residuals.append(float(dv.max()) if dv.size else 0.0)
        V = V_new
    return V, residuals


def pick_action_from_value(
    V: np.ndarray,
    y: int,
    x: int,
    walls_mask: np.ndarray,
    lam: float,
    rng: Optional[np.random.RandomState] = None,
) -> int:
    """
    LMDP-optimal action: sample a neighbor with probability proportional to
    z(u) = exp(V(u)/lam), i.e. a softmax over neighbor values with
    temperature lam -- via Gumbel-max (scale 1.0 gives exact softmax).

    lam doubles as the exploration temperature: lam -> 0 is greedy.
    There is deliberately no separate temperature/momentum/no-backtrack
    machinery here; correct values remove the oscillations those hacks
    patched over.
    """
    H, W = V.shape
    lam = float(max(lam, 1e-6))
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    scores = np.full(4, -1.0e18, dtype=np.float64)
    for i, (dy, dx) in enumerate(dirs):
        yy, xx = y + dy, x + dx
        if 0 <= yy < H and 0 <= xx < W and not walls_mask[yy, xx]:
            scores[i] = float(V[yy, xx]) / lam
    sampler = rng if rng is not None else np.random
    g = sampler.gumbel(loc=0.0, scale=1.0, size=4)
    return int(np.argmax(scores + g))
