"""Information-gain exploration field from belief entropy.

Replaces the two hand-tuned exploration gadgets (Novel + Frontier) with one
principled term: the expected information gained by standing at a cell,
computed from the SAME belief state everything else uses.

Per-cell uncertainty (bits):
    u(v) = H2(p_A(v)) + H2(p_B(v)) + w_map * [unseen and not known-wall]

where H2 is binary entropy of the target beliefs and the w_map term is the
one unresolved bit of map occupancy for never-observed cells. Walls carry
zero uncertainty.

Epistemic reward:
    r_epist(v) = beta_t * meanpool_win(u)(v)

-- "what I would learn if I stood at v", mean-pooled over the observation
window (the window area is folded into beta so reward scale is stable in
win). Injected into the SAME value recursion as target rewards: exploration
and exploitation trade off through one scalar, beta_t, which affect
modulates (curiosity raises it, fear lowers it).

All operations are separable radius-(win//2) shift sums -- no scipy, no
global transforms (locality budget: win//2 per axis, once per tick).
"""

import numpy as np

from .belief import sigmoid


def binary_entropy(p: np.ndarray) -> np.ndarray:
    """H2(p) in bits, safely clipped."""
    p = np.clip(np.asarray(p, dtype=np.float32), 1e-6, 1.0 - 1e-6)
    return (-(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))).astype(np.float32)


def uncertainty_map(
    L_A: np.ndarray,
    L_B: np.ndarray,
    seen: np.ndarray,
    known_walls: np.ndarray,
    w_map: float = 1.0,
) -> np.ndarray:
    """
    Per-cell uncertainty in bits from target beliefs plus map occupancy.

    Unseen cells intentionally carry BOTH their prior-level target entropy
    and the w_map occupancy bit (target uncertainty and map uncertainty are
    different questions a visit would answer).
    """
    u = binary_entropy(sigmoid(L_A)) + binary_entropy(sigmoid(L_B))
    u += np.float32(w_map) * ((~np.asarray(seen, dtype=bool))
                              & (~np.asarray(known_walls, dtype=bool))).astype(np.float32)
    u[np.asarray(known_walls, dtype=bool)] = 0.0
    return u.astype(np.float32)


def pooled_gain(u: np.ndarray, win: int) -> np.ndarray:
    """
    Mean of u over a win x win window centered at each cell (edges use the
    in-bounds portion). Separable shift sums: 2*(win//2) adds per axis.
    """
    u = np.asarray(u, dtype=np.float32)
    half = int(win) // 2
    H, W = u.shape

    def axis_boxsum(a, axis):
        s = a.copy()
        cnt = np.ones_like(a, dtype=np.float32)
        for d in range(1, half + 1):
            sl_fwd = [slice(None)] * 2
            sl_bwd = [slice(None)] * 2
            sl_fwd[axis] = slice(d, None)
            sl_bwd[axis] = slice(None, -d)
            s[tuple(sl_fwd)] += a[tuple(sl_bwd)]
            cnt[tuple(sl_fwd)] += 1.0
            s[tuple(sl_bwd)] += a[tuple(sl_fwd)]
            cnt[tuple(sl_bwd)] += 1.0
        return s, cnt

    s, c1 = axis_boxsum(u, 0)
    s, c2 = axis_boxsum(s, 1)
    # counts compose separably as an outer product of per-axis counts
    cnt = c1 * c2  # c1 varies along axis 0 only, c2 along axis 1 only
    return (s / cnt).astype(np.float32)


def epistemic_beta(
    beta_base: float,
    arousal: float = 0.0,
    pain: float = 0.0,
    k_curiosity: float = 0.5,
    k_fear: float = 0.8,
) -> float:
    """
    Affect-modulated exploration weight: curiosity (arousal) raises it,
    fear (pain) suppresses it. A hurt agent stops sightseeing.
    """
    beta = beta_base * (1.0 + k_curiosity * arousal) * (1.0 - k_fear * pain)
    return float(max(0.0, beta))
