"""Field operations for chemotaxis and navigation."""

import numpy as np
from .diffusion import diffuse_masked


def update_visit_trail(V: np.ndarray, y: int, x: int, walls_mask: np.ndarray,
                       v_decay: float = 0.012, v_diff: float = 0.10, v_inj: float = 1.0) -> np.ndarray:
    """
    Update visit trail field with agent's current position.
    
    Args:
        V: Visit trail field
        y, x: Agent position
        walls_mask: Boolean mask of walls
        v_decay: Trail decay rate
        v_diff: Trail diffusion coefficient
        v_inj: Trail injection strength
        
    Returns:
        Updated visit trail field
    """
    V = V.astype(np.float32, copy=True)
    V *= (1.0 - v_decay)
    V[y, x] = max(V[y, x], v_inj)
    V = diffuse_masked(V, walls_mask, diff=v_diff, decay=0.004, steps=1)
    return V


def update_novelty(N: np.ndarray, pred_err_scalar: float, y: int, x: int, walls_mask: np.ndarray,
                   n_decay: float = 0.02, n_diff: float = 0.06) -> np.ndarray:
    """
    Update novelty field based on prediction error.
    
    Args:
        N: Novelty field
        pred_err_scalar: Prediction error magnitude
        y, x: Agent position
        walls_mask: Boolean mask of walls
        n_decay: Novelty decay rate
        n_diff: Novelty diffusion coefficient
        
    Returns:
        Updated novelty field
    """
    N = N.astype(np.float32, copy=True)
    N *= (1.0 - n_decay)
    N[y, x] = max(N[y, x], float(pred_err_scalar))
    N = diffuse_masked(N, walls_mask, diff=n_diff, decay=0.004, steps=1)
    return N


def corner_hazard(walls_mask: np.ndarray) -> np.ndarray:
    """
    Compute corner hazard map (cells with 2+ adjacent walls).
    
    Args:
        walls_mask: Boolean mask of walls
        
    Returns:
        Corner hazard field (1.0 where corners detected)
    """
    W = (walls_mask > 0).astype(np.float32)
    nn = np.zeros_like(W, dtype=np.float32)
    nn[1:, :]  += W[:-1, :]
    nn[:-1, :] += W[1:,  :]
    nn[:, 1:]  += W[:, :-1]
    nn[:, :-1] += W[:, 1: ]
    return (nn >= 2).astype(np.float32)


def effective_potential(GA, GB, N, Vtrail, Hc,
                        wA=1.0, wB=0.9, wN=0.5, kV=0.7, kH=0.55):
    """
    Compute effective potential field combining all influences.
    
    Args:
        GA: Target A scent field (attractive)
        GB: Target B scent field (attractive)
        N: Novelty field (attractive)
        Vtrail: Visit trail (repulsive)
        Hc: Corner hazard (repulsive)
        wA, wB, wN: Attraction weights
        kV, kH: Repulsion weights
        
    Returns:
        Effective potential field
    """
    return (wA * GA + wB * GB + wN * N) - (kV * Vtrail + kH * Hc)


def pick_action_from_potential(P: np.ndarray, y: int, x: int, walls_mask: np.ndarray,
                               temperature: float = 0.0) -> int:
    """
    Select action based on potential field gradient.
    
    Args:
        P: Potential field
        y, x: Current position
        walls_mask: Boolean mask of walls
        temperature: Softmax temperature (0 = greedy)
        
    Returns:
        Action index (0=up, 1=down, 2=left, 3=right)
    """
    H, W = P.shape
    dirs = [(-1,0), (1,0), (0,-1), (0,1)]
    scores = []
    
    for dy, dx in dirs:
        yy, xx = y + dy, x + dx
        if (yy < 0) or (yy >= H) or (xx < 0) or (xx >= W) or walls_mask[yy, xx]:
            scores.append(-1e9)
        else:
            scores.append(P[yy, xx])
            
    if temperature and temperature > 0:
        s = np.array(scores, dtype=np.float32)
        s -= s.max()
        p = np.exp(s / temperature)
        sm = p.sum()
        if sm <= 0:
            return int(np.argmax(scores))
        p /= sm
        return int(np.random.choice(4, p=p))
    else:
        return int(np.argmax(scores))