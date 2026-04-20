# tests/test_analysis.py
import numpy as np
import pytest

from efi.evaluation.analysis import (
    bfs_shortest_path_len,
    oracle_route_len_through_points,
    compute_coverage,
    compute_frontier_efficiency,
    compute_backtrack_rate,
    compute_path_optimality,
)


def test_bfs_shortest_path_basic():
    # 5x5 empty grid (False = free, True = wall)
    walls = np.zeros((5, 5), dtype=bool)
    assert bfs_shortest_path_len(walls, (0, 0), (0, 0)) == 0
    # Manhattan distance in empty grid with 4-neighbors
    assert bfs_shortest_path_len(walls, (0, 0), (0, 4)) == 4
    assert bfs_shortest_path_len(walls, (0, 0), (4, 4)) == 8


def test_bfs_shortest_path_blocked_and_unreachable():
    walls = np.zeros((5, 5), dtype=bool)
    # Put a solid vertical wall in the middle except a gap
    walls[:, 2] = True
    walls[2, 2] = False  # gap at (2,2)

    # Reachable through the gap (2,0) -> (2,1) -> (2,2) -> (2,3) -> (2,4) = 4 steps
    assert bfs_shortest_path_len(walls, (2, 0), (2, 4)) == 4

    # Fully blocked: seal the gap
    walls[2, 2] = True
    assert bfs_shortest_path_len(walls, (2, 0), (2, 4)) is None


def test_oracle_route_len_through_points():
    walls = np.zeros((5, 5), dtype=bool)
    start = (0, 0)
    points = [(0, 4), (4, 4)]
    # 0,0 -> 0,4 (4) ; 0,4 -> 4,4 (4) => total 8
    assert oracle_route_len_through_points(walls, start, points) == 8

    # Make second leg unreachable by completely surrounding (4,4)
    walls[3, 3:5] = True
    walls[4, 3] = True
    assert oracle_route_len_through_points(walls, start, points) is None


def test_compute_coverage():
    walls = np.zeros((4, 4), dtype=bool)
    walls[0, 0] = True  # one wall
    visited = np.zeros_like(walls, dtype=bool)
    # Visit three distinct free cells
    visited[0, 1] = True
    visited[1, 1] = True
    visited[3, 3] = True
    free = (~walls).sum()  # 15 free cells
    cov = compute_coverage(visited & (~walls), walls)
    assert np.isclose(cov, 3 / free)


def test_compute_frontier_efficiency():
    # Steps: discovered new cells counts and novelty values
    new_cells = [0, 2, 1, 3, 0, 4]
    novelty =   [0.2, 0.7, 0.8, 0.5, 0.1, 0.95]
    # High-novelty (>=0.6) steps: indices 1,2,5 -> (2 + 1 + 4) / 3 = 7/3
    fe = compute_frontier_efficiency(new_cells, novelty, novelty_thresh=0.6)
    assert np.isclose(fe, 7/3)

    # If no high-novelty steps -> 0.0
    novelty_none = [0.1, 0.2, 0.3]
    fe2 = compute_frontier_efficiency([1, 1, 1], novelty_none, novelty_thresh=0.6)
    assert fe2 == 0.0


def test_compute_backtrack_rate():
    # Motions: right, right, left (backtrack), up, down (backtrack)
    motion = [(0, 1), (0, 1), (0, -1), (-1, 0), (1, 0)]
    # Compare consecutive pairs:
    # (0,1)->(0,1) not backtrack ; (0,1)->(0,-1) backtrack
    # (0,-1)->(-1,0) not ; (-1,0)->(1,0) backtrack
    # 2 backtracks over 4 comparisons => 0.5
    br = compute_backtrack_rate(motion)
    assert np.isclose(br, 0.5)

    # Too short -> 0
    assert compute_backtrack_rate([(0, 1)]) == 0.0
    assert compute_backtrack_rate([]) == 0.0


def test_compute_path_optimality_simple():
    walls = np.zeros((5, 5), dtype=bool)
    start = (0, 0)
    pickups = [(0, 4), (4, 4)]
    oracle_len = 8  # as in earlier test
    steps_taken = 12
    opt = compute_path_optimality(walls, start, pickups, steps_taken)
    assert np.isclose(opt, steps_taken / oracle_len)

    # If no pickups -> None
    assert compute_path_optimality(walls, start, [], steps_taken) is None

    # Make route impossible -> None
    walls[:, 2] = True  # solid barrier
    assert compute_path_optimality(walls, start, [(0, 4)], steps_taken) is None