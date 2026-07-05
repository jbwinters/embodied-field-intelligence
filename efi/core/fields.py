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


def compute_reachable_frontier(seen: np.ndarray, known_walls: np.ndarray, 
                              y: int, x: int) -> np.ndarray:
    """
    Compute frontier field only for reachable unseen areas.
    
    Uses flood-fill from current position to find connected free space,
    then computes frontier only within that reachable region.
    
    Args:
        seen: Boolean mask of seen cells
        known_walls: Boolean mask of known walls
        y, x: Current agent position
        
    Returns:
        Reachability-aware frontier field
    """
    from collections import deque
    
    H, W = seen.shape
    
    # Flood-fill to find reachable area from current position
    reachable = np.zeros_like(seen, dtype=bool)
    visited = np.zeros_like(seen, dtype=bool)
    queue = deque([(y, x)])
    visited[y, x] = True
    reachable[y, x] = True
    
    while queue:
        cy, cx = queue.popleft()
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = cy + dy, cx + dx
            if (0 <= ny < H and 0 <= nx < W and 
                not visited[ny, nx] and not known_walls[ny, nx] and seen[ny, nx]):
                # Only traverse through seen areas
                visited[ny, nx] = True
                reachable[ny, nx] = True
                queue.append((ny, nx))
    
    # Now find frontier cells: unseen cells adjacent to reachable seen areas
    frontier_mask = np.zeros_like(seen, dtype=np.float32)
    for cy in range(H):
        for cx in range(W):
            if not seen[cy, cx] and not known_walls[cy, cx]:
                # Check if adjacent to any reachable cell
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = cy + dy, cx + dx
                    if (0 <= ny < H and 0 <= nx < W and reachable[ny, nx]):
                        frontier_mask[cy, cx] = 1.0
                        break
    
    # Diffuse to create smooth frontier field
    frontier = diffuse_masked(frontier_mask, known_walls, 
                            diff=0.15, decay=0.01, steps=3)
    
    return frontier


def wall_proximity_field(walls_mask: np.ndarray, radius: float = 1.0) -> np.ndarray:
    """
    Create a repulsive field near walls to prevent wall-hugging.
    
    Args:
        walls_mask: Boolean mask of walls  
        radius: How far the repulsion extends from walls
        
    Returns:
        Wall proximity penalty field (higher values near walls)
    """
    # Create proximity by distance transform approach
    # Start with 1.0 at non-wall cells adjacent to walls
    H, W = walls_mask.shape
    W_prox = np.zeros((H, W), dtype=np.float32)
    
    # Find cells adjacent to walls
    for y in range(H):
        for x in range(W):
            if not walls_mask[y, x]:  # If not a wall
                # Check if adjacent to any wall
                adjacent_to_wall = False
                for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and walls_mask[ny, nx]:
                        adjacent_to_wall = True
                        break
                if adjacent_to_wall:
                    W_prox[y, x] = 1.0
    
    # Diffuse to create gradient (but not into walls)
    from .diffusion import diffuse_masked
    steps = max(1, int(radius * 2))
    for _ in range(steps):
        W_prox = diffuse_masked(W_prox, walls_mask, diff=0.25, decay=0.1, steps=1)
    
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
    momentum: float = 0.0,
    rng: "np.random.RandomState | None" = None
) -> int:
    """
    Choose among 4-neighbors. If temperature>0, add Gumbel noise to scores.
    Optional momentum toward last_action; optional no-backtrack when stuck.
    Pass a seeded `rng` for reproducible sampling; falls back to the global
    numpy RNG otherwise.
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
        sampler = rng if rng is not None else np.random
        g = sampler.gumbel(loc=0.0, scale=float(temperature), size=4).astype(np.float32)
        return int(np.argmax(s + g))
    else:
        return int(np.argmax(s))