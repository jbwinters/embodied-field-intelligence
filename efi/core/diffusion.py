"""Diffusion operations for CA fields."""

import numpy as np


def diffuse_masked(F: np.ndarray, walls_mask: np.ndarray,
                   diff: float = 0.16, decay: float = 0.01, steps: int = 1) -> np.ndarray:
    """
    Edge-safe, wall-aware diffusion that BLOCKS flow across walls.
    Only passable neighbors contribute to passable destinations.
    
    Non-dimensionalized units:
    - Time unit: Δt = 1 (one CA step)
    - Space unit: Δx = 1 (one grid cell)
    - Diffusion coefficient: D = diff * Δx² / (4*Δt) 
    - CFL constraint: diff ≤ 0.25 for stability in 2D
    - Decay rate: λ = -ln(1-decay) / Δt ≈ decay for small values
    
    Args:
        F: Field to diffuse
        walls_mask: Boolean mask of walls (True = wall)
        diff: Diffusion rate ∈ [0, 0.25] (clamped for stability)
        decay: Exponential decay rate per timestep
        steps: Number of diffusion iterations
        
    Returns:
        Diffused field with walls as zero-flux boundaries
    """
    F = F.astype(np.float32, copy=True)
    W = (walls_mask > 0)
    P = (~W).astype(np.float32)           # 1=passable, 0=wall
    
    # Enforce CFL stability constraint for 2D explicit diffusion
    diff = np.clip(diff, 0.0, 0.25)

    for _ in range(max(1, steps)):
        S = np.zeros_like(F, dtype=np.float32)
        C = np.zeros_like(F, dtype=np.float32)

        # up neighbor contributes into [1:, :]
        S[1:, :]  += F[:-1, :] * P[:-1, :]
        C[1:, :]  += P[:-1, :]
        # down neighbor contributes into [:-1, :]
        S[:-1, :] += F[1:,  :] * P[1:,  :]
        C[:-1, :] += P[1:,  :]
        # left neighbor contributes into [:, 1:]
        S[:, 1:]  += F[:, :-1] * P[:, :-1]
        C[:, 1:]  += P[:, :-1]
        # right neighbor contributes into [:, :-1]
        S[:, :-1] += F[:, 1: ] * P[:, 1: ]
        C[:, :-1] += P[:, 1: ]

        avg = np.divide(S, C, out=np.zeros_like(F), where=(C > 0))
        F = (1.0 - diff) * F + diff * avg
        F *= (1.0 - decay)

        # impose walls as sinks (but NOT borders - agents can walk on edges!)
        F[W] = 0.0
    return F