"""Diffusion operations for CA fields."""

import numpy as np


def diffuse_masked(F: np.ndarray, walls_mask: np.ndarray,
                   diff: float = 0.16, decay: float = 0.01, steps: int = 1) -> np.ndarray:
    """
    Edge-safe, wall-aware diffusion.
    
    Features:
    - 4-neighbors only
    - Normalized by existing neighbors
    - Walls remain 0 and block flow
    - 1-cell border zero to stop corner pooling
    
    Args:
        F: Field to diffuse
        walls_mask: Boolean mask where True indicates walls
        diff: Diffusion coefficient
        decay: Decay rate per step
        steps: Number of diffusion steps
        
    Returns:
        Diffused field
    """
    F = F.astype(np.float32, copy=True)
    W = (walls_mask > 0)
    H, Wd = F.shape

    for _ in range(max(1, steps)):
        S = np.zeros_like(F, dtype=np.float32)
        C = np.zeros_like(F, dtype=np.float32)

        S[1:, :]  += F[:-1, :] ; C[1:, :]  += 1
        S[:-1, :] += F[1:,  :] ; C[:-1, :] += 1
        S[:, 1:]  += F[:, :-1] ; C[:, 1:]  += 1
        S[:, :-1] += F[:, 1: ] ; C[:, :-1] += 1

        avg = np.divide(S, C, out=np.zeros_like(F), where=(C > 0))
        F = (1.0 - diff) * F + diff * avg
        F *= (1.0 - decay)

        F[(walls_mask > 0)] = 0.0
        F[0, :] = F[-1, :] = 0.0
        F[:, 0] = F[:, -1] = 0.0
        
    return F