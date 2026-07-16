"""Tests for local max-plus distance relaxation (Task 6)."""

import sys
from collections import deque

import numpy as np
import pytest

from efi.core.localdist import BIG, dilate1, maxplus_distance


def bfs_reference(sources, blocked):
    """Reference BFS distances (4-connected)."""
    H, W = sources.shape
    D = np.full((H, W), float(BIG))
    q = deque()
    for (y, x) in np.argwhere(sources):
        if blocked is None or not blocked[y, x]:
            D[y, x] = 0
            q.append((y, x))
    while q:
        y, x = q.popleft()
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if (0 <= ny < H and 0 <= nx < W and D[ny, nx] >= BIG
                    and (blocked is None or not blocked[ny, nx])):
                D[ny, nx] = D[y, x] + 1
                q.append((ny, nx))
    return D


class TestMaxplusDistance:
    def test_equivalence_with_bfs_on_random_maps(self):
        rng = np.random.RandomState(0)
        for _ in range(30):
            H, W = 14, 17
            blocked = rng.rand(H, W) < 0.2
            sources = np.zeros((H, W), dtype=bool)
            free = np.argwhere(~blocked)
            for (y, x) in free[rng.choice(len(free), size=2, replace=False)]:
                sources[y, x] = True
            D = maxplus_distance(sources, blocked, iters=H * W)
            D_ref = bfs_reference(sources, blocked)
            np.testing.assert_array_equal(D, D_ref.astype(np.float32))

    def test_light_cone_truncation(self):
        """With iters=k, every cell at true distance > k still holds BIG:
        information cannot outrun the iteration budget."""
        H, W = 21, 21
        sources = np.zeros((H, W), dtype=bool)
        sources[10, 10] = True
        for k in [1, 3, 7]:
            D = maxplus_distance(sources, None, iters=k)
            D_ref = bfs_reference(sources, None)
            assert (D[D_ref > k] >= BIG).all()
            np.testing.assert_array_equal(D[D_ref <= k], D_ref[D_ref <= k].astype(np.float32))

    def test_blocked_cells_do_not_propagate(self):
        H, W = 9, 9
        sources = np.zeros((H, W), dtype=bool)
        sources[4, 1] = True
        blocked = np.zeros((H, W), dtype=bool)
        blocked[:, 4] = True  # full vertical wall
        D = maxplus_distance(sources, blocked, iters=H * W)
        assert (D[:, 5:] >= BIG).all()  # nothing crosses the wall
        assert (D[blocked] >= BIG).all()

    def test_early_convergence_matches_full_run(self):
        rng = np.random.RandomState(1)
        blocked = rng.rand(10, 10) < 0.15
        sources = ~blocked & (rng.rand(10, 10) < 0.05)
        if not sources.any():
            sources[0, 0] = ~blocked[0, 0]
        D1 = maxplus_distance(sources, blocked, iters=100)
        D2 = maxplus_distance(sources, blocked, iters=10000)
        np.testing.assert_array_equal(D1, D2)

    def test_dilate1(self):
        m = np.zeros((5, 5), dtype=bool)
        m[2, 2] = True
        d = dilate1(m)
        assert d.sum() == 5
        assert d[2, 2] and d[1, 2] and d[3, 2] and d[2, 1] and d[2, 3]


class TestMembraneParity:
    def test_same_shell_topology_as_edt(self):
        """New geodesic membrane vs old Euclidean version: identical
        forbidden sets at the default barrier threshold, and support
        mismatches confined to the shell boundary."""
        scipy_ndimage = pytest.importorskip("scipy.ndimage")  # dev-only reference
        from efi.core.membrane import peripersonal_field

        rng = np.random.RandomState(2)
        for _ in range(10):
            H, W = 15, 15
            walls = rng.rand(H, W) < 0.12
            seen = np.ones((H, W), dtype=bool)
            R_t = 2.5  # a fixed radius via base and no affect
            new = peripersonal_field(walls, seen, 7, 7, R_base=R_t)

            dist = scipy_ndimage.distance_transform_edt(~walls)
            old = np.zeros((H, W), dtype=np.float32)
            m = dist < R_t
            old[m] = (1.0 - dist[m] / R_t) ** 2

            # Exact barrier set (threshold 0.75) identical: only wall cells
            # themselves reach that level at this radius
            np.testing.assert_array_equal(new >= 0.75, old >= 0.75)

            # Support mismatch only near the shell boundary (L1 vs L2)
            mismatch = (new > 0) != (old > 0)
            if mismatch.any():
                assert (np.abs(dist[mismatch] - R_t) <= 1.5).all()

    def test_membrane_module_does_not_import_scipy(self):
        """The agent's internals must be scipy-free (locality budget)."""
        import importlib

        for mod in list(sys.modules):
            if mod.startswith("scipy"):
                del sys.modules[mod]
        import efi.core.membrane as membrane
        importlib.reload(membrane)
        assert not any(m.startswith("scipy") for m in sys.modules), \
            "efi.core.membrane pulled in scipy"


class TestFrontierParity:
    def test_frontier_matches_floodfill_semantics(self):
        """New relaxation-based frontier: positive exactly on unseen
        passable cells adjacent to seen space reachable from the agent."""
        from efi.core.fields import compute_reachable_frontier

        rng = np.random.RandomState(3)
        H, W = 13, 13
        walls = rng.rand(H, W) < 0.15
        seen = np.zeros((H, W), dtype=bool)
        seen[4:9, 4:9] = True
        y, x = 6, 6
        walls[y, x] = False
        seen[y, x] = True

        frontier = compute_reachable_frontier(seen, walls, y, x)

        # Reference: BFS through seen & passable
        sources = np.zeros((H, W), dtype=bool)
        sources[y, x] = True
        blocked = walls | ~seen
        blocked[y, x] = False
        D = bfs_reference(sources, blocked)
        reachable = D < BIG
        expected_mask = np.zeros((H, W), dtype=bool)
        for cy in range(H):
            for cx in range(W):
                if not seen[cy, cx] and not walls[cy, cx]:
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < H and 0 <= nx < W and reachable[ny, nx]:
                            expected_mask[cy, cx] = True
                            break

        # The diffused field must be strictly positive at every expected
        # frontier cell (seed cells survive 3 diffusion steps)
        assert (frontier[expected_mask] > 0).all()
