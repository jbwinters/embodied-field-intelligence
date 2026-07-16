"""Tests for LMDP desirability-field value computation (Task 2)."""

from collections import deque

import numpy as np
import pytest

from efi.core.desirability import (
    VBIG,
    lse_neighbor_avg,
    value_sweeps,
    pick_action_from_value,
)


def bfs_distances(walls, sy, sx):
    """Reference BFS geodesic distances (4-connected)."""
    H, W = walls.shape
    D = np.full((H, W), np.inf)
    if walls[sy, sx]:
        return D
    D[sy, sx] = 0
    q = deque([(sy, sx)])
    while q:
        y, x = q.popleft()
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and not walls[ny, nx] and np.isinf(D[ny, nx]):
                D[ny, nx] = D[y, x] + 1
                q.append((ny, nx))
    return D


def run_to_convergence(shape, walls, reward_cell, r, c, lam, sweeps):
    H, W = shape
    q = np.full((H, W), c, dtype=np.float32)
    R = np.full((H, W), -VBIG, dtype=np.float32)
    R[reward_cell] = r
    V = np.zeros((H, W), dtype=np.float32)
    V, _ = value_sweeps(V, q, R, walls, lam=lam, sweeps=sweeps)
    return V


class TestValueSweeps:
    def test_corridor_value_decays_linearly_with_distance(self):
        """Log-domain readout: V(d) is LINEAR in distance with slope
        -(c + lam*log(deg)) -- state cost plus the LMDP's KL control cost
        per step (entropy-regularized shortest path). The point: linear,
        not exponential, attenuation."""
        H, W = 3, 30
        walls = np.zeros((H, W), dtype=bool)
        r, c, lam = 1.0, 0.02, 0.02
        V = run_to_convergence((H, W), walls, (1, W - 1), r, c, lam=lam, sweeps=400)
        ds = np.arange(2, 26)
        vs = np.array([float(V[1, W - 1 - d]) for d in ds])
        slope, intercept = np.polyfit(ds, vs, 1)
        # Linearity: residuals from the fit are tiny
        fit = slope * ds + intercept
        assert float(np.abs(vs - fit).max()) < 0.01
        # Slope is bounded by [pure state cost, cost + max KL control cost]
        # (the exact constant depends on transverse modes of the corridor)
        assert c <= -slope <= c + 2.0 * lam * np.log(4.0)
        # Intercept recovers the injected reward magnitude
        assert intercept == pytest.approx(r, abs=0.15)
        # THE scaling property: the gradient does not attenuate with range.
        # (Diffused scent would have decayed by e^{-kd} here, k ~ O(1).)
        g_near = abs(float(V[1, W - 4]) - float(V[1, W - 5]))
        g_far = abs(float(V[1, W - 25]) - float(V[1, W - 26]))
        assert 0.5 <= g_far / g_near <= 2.0

    def test_greedy_on_V_matches_bfs_shortest_path(self):
        """lam -> 0 limit: greedy next-step is a BFS shortest-path step."""
        rng = np.random.RandomState(7)
        for trial in range(20):
            H = Wd = 15
            walls = rng.rand(H, Wd) < 0.15
            walls[0, :] = walls[-1, :] = walls[:, 0] = walls[:, -1] = False
            free = np.argwhere(~walls)
            ty, tx = free[rng.choice(len(free))]
            D = bfs_distances(walls, ty, tx)
            reachable = np.argwhere(np.isfinite(D) & ~walls)
            sy, sx = reachable[rng.choice(len(reachable))]
            if D[sy, sx] == 0:
                continue
            V = run_to_convergence((H, Wd), walls, (ty, tx), 1.0, 0.01,
                                   lam=0.02, sweeps=H * Wd)
            # Greedy neighbor choice must reduce BFS distance optimally
            dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            neigh_V, neigh_D = [], []
            for dy, dx in dirs:
                ny, nx = sy + dy, sx + dx
                if 0 <= ny < H and 0 <= nx < Wd and not walls[ny, nx]:
                    neigh_V.append(V[ny, nx])
                    neigh_D.append(D[ny, nx])
            chosen = int(np.argmax(neigh_V))
            assert neigh_D[chosen] == min(neigh_D), f"trial={trial}"

    def test_no_value_leaks_through_enclosing_ring(self):
        """Cells outside a solid wall ring receive nothing from the reward
        inside: their values are identical with and without the reward."""
        H, W = 15, 15
        walls = np.zeros((H, W), dtype=bool)
        walls[4, 4:11] = walls[10, 4:11] = True
        walls[4:11, 4] = walls[4:11, 10] = True
        q = np.full((H, W), 0.05, dtype=np.float32)

        R_with = np.full((H, W), -VBIG, dtype=np.float32)
        R_with[7, 7] = 1.0  # inside the ring
        R_without = np.full((H, W), -VBIG, dtype=np.float32)

        V_with, _ = value_sweeps(np.zeros((H, W), np.float32), q, R_with, walls, 0.05, 300)
        V_without, _ = value_sweeps(np.zeros((H, W), np.float32), q, R_without, walls, 0.05, 300)

        outside = np.ones((H, W), dtype=bool)
        outside[4:11, 4:11] = False
        np.testing.assert_allclose(V_with[outside], V_without[outside], atol=1e-3)
        assert V_with[7, 6] > 0.5  # sanity: value did propagate inside

    def test_warm_start_equals_cold_start_on_static_map(self):
        """3 sweeps x 50 ticks warm-started == 150 sweeps cold (same operator)."""
        H, W = 12, 12
        rng = np.random.RandomState(3)
        walls = rng.rand(H, W) < 0.1
        q = np.full((H, W), 0.03, dtype=np.float32)
        R = np.full((H, W), -VBIG, dtype=np.float32)
        R[2, 9] = 1.0
        walls[2, 9] = False

        V_warm = np.zeros((H, W), dtype=np.float32)
        for _ in range(50):
            V_warm, _ = value_sweeps(V_warm, q, R, walls, lam=0.1, sweeps=3)
        V_cold, _ = value_sweeps(np.zeros((H, W), np.float32), q, R, walls, lam=0.1, sweeps=150)

        passable = ~walls
        np.testing.assert_allclose(V_warm[passable], V_cold[passable], rtol=0.01, atol=1e-4)

    def test_walls_hold_sentinel(self):
        H, W = 8, 8
        walls = np.zeros((H, W), dtype=bool)
        walls[3, 3] = True
        q = np.full((H, W), 0.02, dtype=np.float32)
        R = np.full((H, W), -VBIG, dtype=np.float32)
        R[0, 0] = 1.0
        V, _ = value_sweeps(np.zeros((H, W), np.float32), q, R, walls, 0.1, 50)
        assert V[3, 3] == -VBIG

    def test_residuals_decrease(self):
        H, W = 10, 10
        walls = np.zeros((H, W), dtype=bool)
        q = np.full((H, W), 0.02, dtype=np.float32)
        R = np.full((H, W), -VBIG, dtype=np.float32)
        R[5, 5] = 1.0
        _, res = value_sweeps(np.zeros((H, W), np.float32), q, R, walls, 0.1, 200)
        # After the initial injection transient, residuals must shrink
        assert res[-1] < 1e-3
        assert res[-1] < res[5]


class TestActionFromValue:
    def test_greedy_at_small_lam(self):
        """At tiny lam the softmax is effectively greedy."""
        V = np.zeros((5, 5), dtype=np.float32)
        V[1, 2] = 5.0  # up neighbor of (2,2) is best
        walls = np.zeros((5, 5), dtype=bool)
        rng = np.random.RandomState(0)
        actions = [pick_action_from_value(V, 2, 2, walls, lam=0.01, rng=rng)
                   for _ in range(100)]
        assert all(a == 0 for a in actions)

    def test_softmax_proportions_at_moderate_lam(self):
        """Sampling frequencies follow exp(V/lam) ratios."""
        V = np.zeros((5, 5), dtype=np.float32)
        V[1, 2] = 0.7  # up
        V[3, 2] = 0.0  # down
        V[2, 1] = 0.7  # left
        V[2, 3] = 0.7  # right
        walls = np.zeros((5, 5), dtype=bool)
        rng = np.random.RandomState(1)
        counts = np.zeros(4)
        for _ in range(4000):
            counts[pick_action_from_value(V, 2, 2, walls, lam=0.7, rng=rng)] += 1
        # exp(1)/(3*exp(1)+1) ~ 0.294 for each of the three good moves,
        # 1/(3*exp(1)+1) ~ 0.109 for down
        assert counts[1] / 4000 == pytest.approx(0.109, abs=0.03)
        assert counts[0] / 4000 == pytest.approx(0.294, abs=0.04)

    def test_never_selects_wall(self):
        V = np.full((5, 5), 1.0, dtype=np.float32)
        walls = np.zeros((5, 5), dtype=bool)
        walls[1, 2] = walls[3, 2] = walls[2, 1] = True  # only right open
        rng = np.random.RandomState(2)
        for _ in range(50):
            assert pick_action_from_value(V, 2, 2, walls, lam=2.0, rng=rng) == 3
