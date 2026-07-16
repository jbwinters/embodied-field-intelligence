"""Tests for non-stationary ForageWorld and regret metrics (Task 10)."""

import numpy as np
import pytest

from efi.configs import EnvConfig
from efi.envs import ForageWorld
from efi.evaluation.metrics import adaptation_lag, moving_average, regret_series, regret_slopes


def drive(env, actions):
    out = []
    for a in actions:
        out.append(env.step(a))
    return out


class TestDeterminism:
    def test_same_seed_same_schedules(self):
        """Identical seeds + identical actions => identical worlds through
        regrow, drift, and swap events."""
        cfg = EnvConfig(H=13, W=13, p_regrow=0.05, T_shift=30, T_swap=60,
                        max_steps=200, seed=9)
        rng = np.random.RandomState(0)
        actions = list(rng.randint(0, 4, size=150))

        def trace(seed_offset=0):
            env = ForageWorld(cfg)
            env.reset()
            states = []
            for a in actions:
                _, r, done, info = env.step(a)
                states.append((r, env.y, env.x, env.TA.sum(), env.TB.sum(),
                               env.reward_A, env.reward_B))
                if done:
                    break
            return states

        assert trace() == trace()

    def test_clone_matches_original_under_same_actions(self):
        cfg = EnvConfig(H=13, W=13, p_regrow=0.05, T_shift=25, T_swap=50,
                        max_steps=300, seed=4)
        env = ForageWorld(cfg)
        env.reset()
        rng = np.random.RandomState(1)
        warmup = list(rng.randint(0, 4, size=40))
        drive(env, warmup)

        twin = env.clone()
        tail = list(rng.randint(0, 4, size=100))
        for a in tail:
            o1 = env.step(a)
            o2 = twin.step(a)
            assert o1[1] == o2[1]                     # reward
            assert (env.y, env.x) == (twin.y, twin.x)
            assert np.array_equal(env.TA, twin.TA)
            assert np.array_equal(env.TB, twin.TB)


class TestEvents:
    def test_regrow_respawns_and_never_on_walls(self):
        cfg = EnvConfig(H=11, W=11, p_regrow=0.5, n_targets_A=2, n_targets_B=1,
                        max_steps=400, seed=2)
        env = ForageWorld(cfg)
        env.reset()
        rng = np.random.RandomState(3)
        picked = 0
        for _ in range(400):
            _, _, done, info = env.step(int(rng.randint(4)))
            if info["picked"]:
                picked += 1
            assert not (env.TA & env.walls).any()
            assert not (env.TB & env.walls).any()
            # total live+pending never exceeds the configured counts
            live_pending_A = env.TA.sum() + sum(1 for d, k in env._respawn_queue if k == "A")
            assert live_pending_A <= cfg.n_targets_A
            if done:
                break
        if picked:
            assert env.TA.sum() + env.TB.sum() > 0 or env._respawn_queue

    def test_regrow_keeps_episode_alive(self):
        cfg = EnvConfig(H=9, W=9, p_regrow=0.5, n_targets_A=1, n_targets_B=0,
                        max_steps=150, seed=5)
        env = ForageWorld(cfg)
        env.reset()
        rng = np.random.RandomState(4)
        steps = 0
        done = False
        while not done:
            _, _, done, _ = env.step(int(rng.randint(4)))
            steps += 1
        assert steps == cfg.max_steps  # only the cap ends it

    def test_drift_moves_targets_within_radius(self):
        cfg = EnvConfig(H=15, W=15, T_shift=10, p_move=1.0, r_drift=3,
                        p_wall=0.0, n_targets_A=3, n_targets_B=0,
                        max_steps=100, seed=6)
        env = ForageWorld(cfg)
        env.reset()
        before = np.argwhere(env.TA)
        for _ in range(10):  # cross one shift boundary
            env.step(0)
        after = np.argwhere(env.TA)
        assert len(after) == len(before)
        assert not np.array_equal(np.sort(before, axis=0), np.sort(after, axis=0))
        # Chebyshev containment: every new target within r of SOME old one
        for (ny, nx) in after:
            d = max(abs(int(ny) - int(oy)) for _ in [0] for (oy, ox) in [min(before, key=lambda p: max(abs(int(ny)-int(p[0])), abs(int(nx)-int(p[1]))))]) if len(before) else 0
            cheb = min(max(abs(int(ny) - int(oy)), abs(int(nx) - int(ox))) for (oy, ox) in before)
            assert cheb <= cfg.r_drift

    def test_swap_flips_rewards_at_T_swap(self):
        cfg = EnvConfig(H=9, W=9, T_swap=5, max_steps=50, seed=7)
        env = ForageWorld(cfg)
        env.reset()
        rA0, rB0 = env.reward_A, env.reward_B
        for _ in range(5):
            env.step(0)
        assert env.reward_A == rB0
        assert env.reward_B == rA0

    def test_defaults_change_nothing(self):
        """With all knobs off, behavior identical to the stationary env."""
        cfg = EnvConfig(H=11, W=11, seed=8)
        env = ForageWorld(cfg)
        env.reset()
        rng = np.random.RandomState(5)
        for _ in range(100):
            _, _, done, _ = env.step(int(rng.randint(4)))
            assert env.reward_A == cfg.reward_A
            assert not env._respawn_queue
            if done:
                break


class TestRegretMetrics:
    def test_adaptation_lag_on_synthetic_series(self):
        """Known answer: reward 1.0, drops to 0 at t=100, recovers linearly
        over 50 steps."""
        r = np.ones(300)
        r[100:150] = np.linspace(0.0, 1.0, 50)
        lags = adaptation_lag(r, [100], window=20, baseline_window=50,
                              tolerance=0.2)
        assert lags[0] is not None
        assert 30 <= lags[0] <= 70  # ma(20) recovers as the ramp completes

    def test_adaptation_lag_never_recovers(self):
        r = np.ones(300)
        r[100:] = -1.0
        lags = adaptation_lag(r, [100])
        assert lags[0] is None

    def test_regret_series_and_slopes(self):
        agent = [0.0] * 200
        oracle = [0.1] * 200
        reg = regret_series(agent, oracle)
        assert reg[-1] == pytest.approx(20.0)
        slopes = regret_slopes(reg, window=100)
        assert all(s == pytest.approx(0.1) for s in slopes)

    def test_moving_average_prefix(self):
        ma = moving_average([1, 2, 3, 4], window=2)
        assert ma[0] == 1.0 and ma[1] == 1.5 and ma[3] == 3.5
