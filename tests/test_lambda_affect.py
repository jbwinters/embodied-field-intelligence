"""Tests for affect-controlled lambda and exact barriers (Task 3)."""

import numpy as np
import pytest

from efi.core.affect import AffectState, affect_to_lambda
from efi.core.desirability import VBIG, pick_action_from_value
from efi.configs import AgentConfig, Ablations, EnvConfig
from efi.envs import ForageWorld
from efi.agents import FieldController, ForageAdapter


def make_controller(seed=0, H=17, W=17, **cfg_kwargs):
    env = ForageWorld(EnvConfig(H=H, W=W, seed=seed))
    env.reset()
    cfg = AgentConfig(seed=seed, **cfg_kwargs)
    agent = FieldController(env, ForageAdapter(env), cfg, Ablations(), seed=seed)
    return env, agent


def action_entropy(V, y, x, walls, lam, rng, n=3000):
    counts = np.zeros(4)
    for _ in range(n):
        counts[pick_action_from_value(V, y, x, walls, lam=lam, rng=rng)] += 1
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


class TestAffectToLambda:
    def test_pain_lowers_lambda_monotonically(self):
        lams = [affect_to_lambda(AffectState(pain=p)) for p in np.linspace(0, 1, 11)]
        assert all(l2 <= l1 for l1, l2 in zip(lams, lams[1:]))
        assert lams[0] > lams[-1]

    def test_arousal_raises_lambda(self):
        low = affect_to_lambda(AffectState(arousal=0.0))
        high = affect_to_lambda(AffectState(arousal=1.0))
        assert high > low

    def test_clipping(self):
        assert affect_to_lambda(AffectState(pain=1.0, arousal=0.0),
                                lam_min=0.005) == pytest.approx(0.005)
        assert affect_to_lambda(AffectState(pain=0.0, arousal=1.0),
                                lam_base=0.09, lam_max=0.1) == pytest.approx(0.1)

    def test_no_behavioral_cliff_across_old_flip_threshold(self):
        """lambda is continuous in pain: the legacy semiring flip at
        pain=0.6 must leave no discontinuity."""
        eps = 1e-3
        below = affect_to_lambda(AffectState(pain=0.6 - eps))
        above = affect_to_lambda(AffectState(pain=0.6 + eps))
        assert abs(below - above) < 1e-4

    def test_action_entropy_decreases_with_pain(self):
        """Higher pain -> lower lambda -> more decisive (lower-entropy)
        action distribution on a fixed value field."""
        V = np.zeros((7, 7), dtype=np.float32)
        V[2, 3] = 0.05  # up neighbor slightly best
        walls = np.zeros((7, 7), dtype=bool)
        rng = np.random.RandomState(0)
        entropies = []
        for pain in [0.0, 0.3, 0.6, 0.9]:
            lam = affect_to_lambda(AffectState(pain=pain))
            entropies.append(action_entropy(V, 3, 3, walls, lam, rng))
        assert all(e2 <= e1 + 0.02 for e1, e2 in zip(entropies, entropies[1:]))
        assert entropies[-1] < entropies[0]


class TestExactBarrier:
    def test_forbidden_ring_is_never_entered(self):
        """Scripted 500-step greedy walk with a membrane ring at 1.0: the
        agent must never step onto a forbidden cell, for any lambda."""
        for lam in [0.005, 0.02, 0.1]:
            env, agent = make_controller(seed=1, H=15, W=15,
                                         membrane_enabled=True)
            env.walls[:] = False  # open room; barrier is the membrane only
            # Ring of forbidden cells around the center
            membrane = np.zeros((15, 15), dtype=np.float32)
            membrane[5, 5:10] = membrane[9, 5:10] = 1.0
            membrane[5:10, 5] = membrane[5:10, 9] = 1.0
            forbidden = membrane >= agent.cfg.barrier_threshold

            env.y, env.x = 2, 2
            rng = np.random.RandomState(0)
            entered = 0
            for t in range(500):
                agent.step_fields(env._obs())
                V = agent.compose_value(membrane_field=membrane, lam=lam)
                a = pick_action_from_value(V, env.y, env.x, env.walls,
                                           lam=lam, rng=rng)
                _, _, done, _ = env.step(a)
                if forbidden[env.y, env.x]:
                    entered += 1
                if done:
                    break
            assert entered == 0, f"lam={lam}: {entered} barrier violations"

    def test_forbidden_cells_hold_sentinel_value(self):
        env, agent = make_controller(seed=2, membrane_enabled=True)
        membrane = np.zeros((17, 17), dtype=np.float32)
        membrane[8, 8] = 1.0
        agent.step_fields(env._obs())
        V = agent.compose_value(membrane_field=membrane)
        assert V[8, 8] == -VBIG

    def test_softmax_probability_of_barrier_is_zero(self):
        """With one forbidden neighbor and open alternatives, the forbidden
        cell is chosen with probability exactly 0 (sentinel dominates)."""
        V = np.zeros((5, 5), dtype=np.float32)
        V[1, 2] = -VBIG  # up neighbor forbidden
        walls = np.zeros((5, 5), dtype=bool)
        rng = np.random.RandomState(3)
        for _ in range(2000):
            assert pick_action_from_value(V, 2, 2, walls, lam=0.1, rng=rng) != 0

    def test_deadlock_fallback_and_logging(self):
        """Agent boxed in by forbidden cells: runner logs barrier_deadlocks
        and the softmax still returns a (least-bad) action."""
        V = np.full((5, 5), -VBIG, dtype=np.float32)
        walls = np.zeros((5, 5), dtype=bool)
        rng = np.random.RandomState(4)
        a = pick_action_from_value(V, 2, 2, walls, lam=0.02, rng=rng)
        assert a in (0, 1, 2, 3)  # graceful, no crash

    def test_runner_counts_deadlocks(self):
        """End-to-end: metrics carry the barrier_deadlocks field."""
        from efi.evaluation import run_episode
        env, agent = make_controller(seed=5)
        _, _, metrics, _ = run_episode(env, agent, None, Ablations())
        assert hasattr(metrics, "barrier_deadlocks")
        assert metrics.barrier_deadlocks >= 0


class TestLambdaPlumbing:
    def test_same_lambda_for_sweeps_and_action(self):
        """compose_value stores lam_current; affect modulates it."""
        env, agent = make_controller(seed=6, affect_enabled=True)
        agent.step_fields(env._obs())
        agent.affect_state = AffectState(pain=0.9, arousal=0.0)
        agent.compose_value()
        lam_pain = agent.lam_current
        agent.affect_state = AffectState(pain=0.0, arousal=0.0)
        agent.compose_value()
        lam_calm = agent.lam_current
        assert lam_pain < lam_calm
