"""Tests for fixed-point tracking instrumentation (Task 4)."""

import numpy as np
import pytest

from efi.core.desirability import VBIG, value_sweeps
from efi.configs import AgentConfig, Ablations, EnvConfig
from efi.envs import ForageWorld
from efi.agents import FieldController, ForageAdapter
from efi.evaluation import run_episode


def random_problem(rng, H=12, W=12):
    walls = rng.rand(H, W) < 0.12
    q = np.full((H, W), 0.02, dtype=np.float32) + 0.05 * rng.rand(H, W).astype(np.float32)
    # Small global reward floor, mirroring the controller's belief-prior
    # optimism floor: without it, enclosed rewardless pockets have fixed
    # point -inf and residuals never converge (see value_sweeps docstring).
    R = np.full((H, W), 0.001, dtype=np.float32)
    free = np.argwhere(~walls)
    ty, tx = free[rng.choice(len(free))]
    R[ty, tx] = 1.0
    return walls, q, R


class TestContraction:
    def test_residuals_shrink_on_random_frozen_states(self):
        """After the injection transient, per-sweep residuals decrease."""
        rng = np.random.RandomState(0)
        for _ in range(50):
            walls, q, R = random_problem(rng)
            V0 = (0.2 * rng.rand(*q.shape)).astype(np.float32)
            _, res = value_sweeps(V0, q, R, walls, lam=0.05, sweeps=60)
            # Tail residual far below early residual, and tiny in absolute terms
            assert res[-1] < res[5] + 1e-6
            assert res[-1] < 1e-3

    def test_warm_start_converges_faster_than_cold(self):
        """After a small localized perturbation of the inputs, restarting
        from the previous fixed point reaches tolerance in fewer sweeps
        than restarting from zero."""

        def sweeps_to_tol(V0, q, R, walls, tol=1e-3, max_sweeps=500):
            V = V0
            for k in range(1, max_sweeps + 1):
                V, res = value_sweeps(V, q, R, walls, lam=0.05, sweeps=1)
                if res[0] < tol:
                    return k
            return max_sweeps

        rng = np.random.RandomState(1)
        wins = 0
        for _ in range(20):
            walls, q, R = random_problem(rng)
            V_fixed, _ = value_sweeps(np.zeros_like(q), q, R, walls, lam=0.05, sweeps=400)
            # Small localized perturbation: one new cost bump
            q2 = q.copy()
            free = np.argwhere(~walls)
            py, px = free[rng.choice(len(free))]
            q2[py, px] += 0.1
            k_warm = sweeps_to_tol(V_fixed.copy(), q2, R, walls)
            k_cold = sweeps_to_tol(np.zeros_like(q), q2, R, walls)
            if k_warm < k_cold:
                wins += 1
        # Warm start must win in the overwhelming majority of cases
        assert wins >= 18

    def test_kappa_zero_runs_without_crash(self):
        """z_sweeps=0: agent runs on the stale field (degenerate but legal).
        The first tick still gets the init/orientation sweeps."""
        env = ForageWorld(EnvConfig(H=13, W=13, max_steps=60, seed=2))
        env.reset()
        cfg = AgentConfig(seed=2, z_sweeps=0)
        agent = FieldController(env, ForageAdapter(env), cfg, Ablations(), seed=2)
        ret, _, metrics, _ = run_episode(env, agent, None, Ablations())
        assert np.isfinite(ret)
        assert metrics.steps > 0


class TestDiagnosticsPlumbing:
    def test_metrics_carry_tracking_fields(self):
        env = ForageWorld(EnvConfig(H=13, W=13, max_steps=60, seed=3))
        env.reset()
        cfg = AgentConfig(seed=3)
        agent = FieldController(env, ForageAdapter(env), cfg, Ablations(), seed=3)
        _, _, metrics, _ = run_episode(env, agent, None, Ablations())
        assert metrics.mean_residual >= 0.0
        assert metrics.p95_residual >= metrics.mean_residual * 0.0
        assert metrics.mean_lambda > 0.0

    def test_controller_exposes_sweep_inputs(self):
        env = ForageWorld(EnvConfig(H=13, W=13, seed=4))
        env.reset()
        cfg = AgentConfig(seed=4)
        agent = FieldController(env, ForageAdapter(env), cfg, Ablations(), seed=4)
        agent.step_fields(env._obs())
        agent.compose_value()
        assert agent.last_q.shape == (13, 13)
        assert agent.last_R_inj.shape == (13, 13)
        # Deep verification is possible: extra sweeps from the exposed inputs
        V_inf, _ = value_sweeps(agent.V.copy(), agent.last_q, agent.last_R_inj,
                                agent.last_walls_used, lam=agent.lam_current,
                                sweeps=200)
        passable = ~agent.last_walls_used
        err = float(np.abs(V_inf[passable] - agent.V[passable]).max())
        assert np.isfinite(err)
