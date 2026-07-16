"""Tests for the two-level field pyramid (Task 11)."""

import numpy as np
import pytest

from efi.core.desirability import VBIG, value_sweeps
from efi.core.pyramid import (
    pool_blocked,
    pool_cost,
    pool_reward,
    pyramid_value_sweeps,
    upsample,
)


class TestPooling:
    def test_blocked_iff_all_blocked(self):
        b = np.zeros((4, 4), dtype=bool)
        b[0:2, 0:2] = True          # fully blocked quad
        b[0, 2] = True              # half-open quad
        cb = pool_blocked(b)
        assert cb[0, 0]
        assert not cb[0, 1]
        assert not cb[1, 1]

    def test_reward_max_pool(self):
        R = np.full((4, 4), -VBIG, dtype=np.float32)
        R[1, 1] = 2.0
        cR = pool_reward(R)
        assert cR[0, 0] == pytest.approx(2.0)
        assert cR[1, 1] == -VBIG

    def test_cost_mean_over_passable(self):
        q = np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32)
        blocked = np.array([[False, False], [True, False]])
        cq = pool_cost(q, blocked)
        assert cq[0, 0] == pytest.approx((1 + 3 + 7) / 3)

    def test_upsample_shape_crop(self):
        c = np.arange(6, dtype=np.float32).reshape(2, 3)
        up = upsample(c, (3, 5))
        assert up.shape == (3, 5)
        assert up[0, 0] == 0 and up[2, 4] == 5


class TestConsistency:
    def test_pyramid_init_converges_faster_than_plain_warm_start(self):
        """Same number of FINE sweeps: the coarse-initialized field must be
        closer to the true fixed point (the pyramid's whole point: coarse
        sweeps are cheap horizon)."""
        rng = np.random.RandomState(0)
        wins = 0
        for trial in range(12):
            H = W = 50
            walls = rng.rand(H, W) < 0.12
            q = np.full((H, W), 0.01, dtype=np.float32)
            R = np.full((H, W), 0.001, dtype=np.float32)
            free = np.argwhere(~walls)
            ty, tx = free[rng.choice(len(free))]
            R[ty, tx] = 1.5

            V_ref, _ = value_sweeps(np.zeros((H, W), np.float32), q, R, walls,
                                    lam=0.02, sweeps=500)
            passable = ~walls

            V_plain, _ = value_sweeps(np.zeros((H, W), np.float32), q, R, walls,
                                      lam=0.02, sweeps=10)
            V_pyr, _, _ = pyramid_value_sweeps(np.zeros((H, W), np.float32),
                                               q, R, walls, lam=0.02, sweeps=10,
                                               coarse_sweeps=60)
            err_plain = float(np.abs(V_plain[passable] - V_ref[passable]).mean())
            err_pyr = float(np.abs(V_pyr[passable] - V_ref[passable]).mean())
            if err_pyr < err_plain:
                wins += 1
        assert wins >= 10  # coarse horizon wins on nearly every map

    def test_pyramid_never_lifts_walls(self):
        rng = np.random.RandomState(1)
        H = W = 30
        walls = rng.rand(H, W) < 0.15
        q = np.full((H, W), 0.01, dtype=np.float32)
        R = np.full((H, W), 0.001, dtype=np.float32)
        V, _, _ = pyramid_value_sweeps(np.zeros((H, W), np.float32), q, R,
                                       walls, lam=0.02, sweeps=5)
        assert (V[walls] == -VBIG).all()


class TestBehavior:
    def test_no_regression_on_small_grids(self):
        """15x15: pyramid on vs off within noise."""
        from efi.configs import AgentConfig, Ablations, EnvConfig
        from efi.evaluation import run_experiment
        env_cfg = EnvConfig(H=15, W=15, max_steps=200)
        rets = {}
        for levels in (1, 2):
            cfg = AgentConfig(valA_init=1.0, pyramid_levels=levels)
            r = run_experiment(env_cfg, cfg, None, Ablations(schema=0),
                               episodes=15, seeds=2, use_controller=True)
            rets[levels] = r.mean_return
        assert rets[2] >= rets[1] - 0.4
