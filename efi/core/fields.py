"""Field operations for chemotaxis and navigation."""

import numpy as np
from .diffusion import diffuse_masked


def update_visit_trail(V: np.ndarray, y: int, x: int, walls_mask: np.ndarray,
                       v_decay: float = 0.012, v_diff: float = 0.08, v_inj: float = 1.0,
                       v_cap: float = 3.0) -> np.ndarray:
    """
    Additive deposit with cap; slightly lower diffusion to keep repulsion local.
    """
    V = V.astype(np.float32, copy=True)
    # First inject at current position (before decay)
    V[y, x] = np.minimum(V[y, x] + v_inj, v_cap)  # <-- additive, capped
    # Then apply decay (so current position doesn't immediately decay)
    V *= (1.0 - v_decay)
    # Diffuse to spread the trail
    V = diffuse_masked(V, walls_mask, diff=v_diff, decay=0.002, steps=1)
    return V


def update_novelty(N: np.ndarray, pred_err_scalar: float, y: int, x: int, walls_mask: np.ndarray,
                   n_decay: float = 0.02, n_diff: float = 0.06, gain: float = 6.0) -> np.ndarray:
    """
    Stronger novelty deposit (gain) with clamping to [0,1].
    """
    N = N.astype(np.float32, copy=True)
    N *= (1.0 - n_decay)
    inj = float(gain * pred_err_scalar)
    N[y, x] = min(1.0, max(N[y, x], inj))
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


def wall_proximity_field(walls_mask: np.ndarray, radius: float = 1.0) -> np.ndarray:
    """
    Create a repulsive field near walls to prevent wall-hugging.
    
    Args:
        walls_mask: Boolean mask of walls  
        radius: How far the repulsion extends from walls
        
    Returns:
        Wall proximity penalty field (higher values near walls)
    """
    from .diffusion import diffuse_masked
    
    # Start with walls as source
    W = walls_mask.astype(np.float32)
    
    # Diffuse wall presence to create proximity gradient
    # More steps = wider repulsion zone
    steps = max(1, int(radius * 2))
    W_prox = diffuse_masked(W, walls_mask, diff=0.25, decay=0.05, steps=steps)
    
    # Scale so it's strong near walls but drops off
    W_prox = np.clip(W_prox * 2.0, 0, 1.0)
    
    return W_prox


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


def pick_action_from_potential(
    P: np.ndarray, y: int, x: int, walls_mask: np.ndarray,
    temperature: float = 0.0,
    last_action: int | None = None,
    no_backtrack: bool = False,
    momentum: float = 0.0
) -> int:
    """
    Choose among 4-neighbors. If temperature>0, add Gumbel noise to scores.
    Optional momentum toward last_action; optional no-backtrack when stuck.
    """
    import numpy as np

    H, W = P.shape
    dirs = [(-1,0), (1,0), (0,-1), (0,1)]
    scores = []

    for dy, dx in dirs:
        yy, xx = y + dy, x + dx
        if (yy < 0) or (yy >= H) or (xx < 0) or (xx >= W) or walls_mask[yy, xx]:
            scores.append(-1e9)
        else:
            scores.append(P[yy, xx])

    s = np.array(scores, dtype=np.float32)

    # Momentum: prefer continuing the previous action a bit
    if (momentum > 0.0) and (last_action is not None) and (0 <= last_action < 4):
        s[last_action] += float(momentum)

    # No immediate reverse (helps break ping-pong) when stuck
    if no_backtrack and (last_action is not None):
        reverse = [1, 0, 3, 2][last_action]
        s[reverse] -= 1e3  # hard-penalize exact backstep

    if temperature and temperature > 0:
        # Gumbel-max sampling (local noise in the action space, not the map)
        g = np.random.gumbel(loc=0.0, scale=float(temperature), size=4).astype(np.float32)
        return int(np.argmax(s + g))
    else:
        return int(np.argmax(s))