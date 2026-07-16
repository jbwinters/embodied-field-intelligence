"""Local max-plus distance relaxation (Bellman-Ford / brushfire).

Replaces the two internal-locality violations (scipy's global
distance_transform_edt in the membrane, BFS flood-fill in the frontier)
with iterated radius-1 relaxation:

    D <- min(D, 1 + min over 4-neighbors of D)

This is the tropical (min-plus) semiring analogue of the diffusion the rest
of EFI runs on, and the lam -> 0 limit of the value recursion in
`efi/core/desirability.py`. Distances are L1/geodesic (wall-respecting),
which is MORE correct for membranes than Euclidean EDT: shells hug geometry
through gaps instead of jumping walls.

`iters` bounds the propagation radius explicitly -- this IS the light cone:
one call propagates information at most `iters` cells. Every call site must
pass a deliberate iteration budget.
"""

from typing import Optional

import numpy as np

BIG = np.float32(1.0e6)


def maxplus_distance(
    sources_mask: np.ndarray,
    blocked_mask: Optional[np.ndarray] = None,
    iters: int = 64,
    big: float = float(BIG),
) -> np.ndarray:
    """
    Geodesic (4-connected, L1) distance from `sources_mask` cells, by local
    relaxation only.

    Args:
        sources_mask: boolean, True = distance-0 source cells
        blocked_mask: boolean, True = impassable; blocked cells never update
            and contribute `big` as neighbors (they do not propagate)
        iters: maximum relaxation iterations == maximum propagation radius
            (the light cone). Cells farther than `iters` keep `big`.
        big: sentinel for "unreached"

    Returns:
        float32 (H, W) distances; `big` where unreached/blocked (a blocked
        source stays blocked).
    """
    src = np.asarray(sources_mask) > 0
    H, W = src.shape
    big = np.float32(big)
    D = np.where(src, np.float32(0.0), big).astype(np.float32)

    blocked = None
    if blocked_mask is not None:
        blocked = np.asarray(blocked_mask) > 0
        D[blocked] = big

    for _ in range(max(0, int(iters))):
        nb = np.full((H, W), big, dtype=np.float32)
        # Same 4-shift stencil as diffuse_masked (efi/core/diffusion.py).
        nb[1:, :] = np.minimum(nb[1:, :], D[:-1, :])
        nb[:-1, :] = np.minimum(nb[:-1, :], D[1:, :])
        nb[:, 1:] = np.minimum(nb[:, 1:], D[:, :-1])
        nb[:, :-1] = np.minimum(nb[:, :-1], D[:, 1:])

        D_new = np.minimum(D, 1.0 + nb)
        D_new = np.minimum(D_new, big)  # keep the sentinel exact
        if blocked is not None:
            D_new[blocked] = big
        if np.array_equal(D_new, D):
            break  # converged early; result identical to running all iters
        D = D_new
    return D


def dilate1(mask: np.ndarray) -> np.ndarray:
    """One step of 4-neighbor dilation (radius-1 local operation)."""
    m = np.asarray(mask) > 0
    out = m.copy()
    out[1:, :] |= m[:-1, :]
    out[:-1, :] |= m[1:, :]
    out[:, 1:] |= m[:, :-1]
    out[:, :-1] |= m[:, 1:]
    return out
