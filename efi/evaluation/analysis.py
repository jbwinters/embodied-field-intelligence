"""Analysis helpers for computing advanced metrics."""

from collections import deque
from typing import List, Optional, Tuple
import numpy as np


Coord = Tuple[int, int]


def bfs_shortest_path_len(walls: np.ndarray, start: Coord, goal: Coord) -> Optional[int]:
    """Grid BFS shortest path length (4-neighbors), returns None if unreachable."""
    H, W = walls.shape
    if start == goal:
        return 0
    if walls[start] or walls[goal]:
        return None

    q = deque([start])
    dist = np.full((H, W), -1, dtype=np.int32)
    dist[start] = 0
    while q:
        y, x = q.popleft()
        d = dist[y, x] + 1
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and not walls[ny, nx] and dist[ny, nx] == -1:
                dist[ny, nx] = d
                if (ny, nx) == goal:
                    return int(d)
                q.append((ny, nx))
    return None


def oracle_route_len_through_points(walls: np.ndarray, start: Coord, points: List[Coord]) -> Optional[int]:
    """
    Sum of BFS shortest paths visiting points in their observed order.
    Returns None if any leg is unreachable.
    """
    total = 0
    cur = start
    for p in points:
        d = bfs_shortest_path_len(walls, cur, p)
        if d is None:
            return None
        total += d
        cur = p
    return total


def compute_coverage(visited_free_mask: np.ndarray, walls: np.ndarray) -> float:
    """% of free cells that were actually *visited* at least once."""
    free = (~walls).sum()
    if free == 0:
        return 0.0
    return float(visited_free_mask.sum() / free)


def compute_frontier_efficiency(new_cells_per_step: List[int], novelty_at_steps: List[float],
                                novelty_thresh: float = 0.6) -> float:
    """
    Average 'new cells discovered per step' during steps with high novelty.
    If no high-novelty steps, returns 0.
    """
    num = 0
    den = 0
    for nnew, nov in zip(new_cells_per_step, novelty_at_steps):
        if nov >= novelty_thresh:
            num += int(nnew)
            den += 1
    return float(num / den) if den > 0 else 0.0


def compute_backtrack_rate(motion_history: List[Coord]) -> float:
    """
    Fraction of moves that immediately reverse the previous move.
    motion_history: list of (dy, dx) for each *executed* move, skipping bumps.
    """
    if len(motion_history) < 2:
        return 0.0
    backtracks = 0
    for (dy1, dx1), (dy2, dx2) in zip(motion_history[:-1], motion_history[1:]):
        if (dy1 == -dy2) and (dx1 == -dx2):
            backtracks += 1
    return float(backtracks / max(1, len(motion_history) - 1))


def compute_path_optimality(walls: np.ndarray, start: Coord, pickups_in_order: List[Coord], steps_taken: int
                           ) -> Optional[float]:
    """
    steps_taken / oracle_length_through_pickups.
    Returns None if oracle path is undefined (e.g., no pickups or unreachable).
    """
    if not pickups_in_order:
        return None
    oracle_len = oracle_route_len_through_points(walls, start, pickups_in_order)
    if oracle_len is None or oracle_len == 0:
        return None
    return float(steps_taken) / float(oracle_len)