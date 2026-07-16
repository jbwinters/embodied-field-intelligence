"""Tests for information-gain exploration (Task 5)."""

import numpy as np
import pytest

from efi.core.infogain import (
    binary_entropy,
    epistemic_beta,
    pooled_gain,
    uncertainty_map,
)
from efi.configs import AgentConfig, Ablations, EnvConfig
from efi.envs import ForageWorld
from efi.agents import FieldController, ForageAdapter
from efi.evaluation import run_episode


class TestPrimitives:
    def test_entropy_extremes(self):
        assert binary_entropy(np.float32(0.5)) == pytest.approx(1.0, abs=1e-4)
        assert binary_entropy(np.float32(0.0001)) < 0.01
        assert binary_entropy(np.float32(0.9999)) < 0.01

    def test_uncertainty_zero_when_fully_observed_and_disconfirmed(self):
        H = W = 9
        L = np.full((H, W), -8.0, dtype=np.float32)   # hard "no target"
        seen = np.ones((H, W), dtype=bool)
        walls = np.zeros((H, W), dtype=bool)
        u = uncertainty_map(L, L, seen, walls)
        assert float(u.max()) < 0.01

    def test_unseen_cells_carry_map_bit_plus_prior_entropy(self):
        H = W = 9
        L = np.full((H, W), -4.0, dtype=np.float32)   # prior
        seen = np.zeros((H, W), dtype=bool)
        walls = np.zeros((H, W), dtype=bool)
        u = uncertainty_map(L, L, seen, walls, w_map=1.0)
        expected = 1.0 + 2.0 * float(binary_entropy(np.float32(1 / (1 + np.exp(4.0)))))
        assert u[4, 4] == pytest.approx(expected, abs=0.02)

    def test_walls_carry_zero_uncertainty(self):
        H = W = 9
        L = np.zeros((H, W), dtype=np.float32)        # max entropy
        seen = np.zeros((H, W), dtype=bool)
        walls = np.zeros((H, W), dtype=bool)
        walls[3, 3] = True
        u = uncertainty_map(L, L, seen, walls)
        assert u[3, 3] == 0.0

    def test_pooled_gain_is_window_mean(self):
        u = np.zeros((11, 11), dtype=np.float32)
        u[5, 5] = 25.0
        g = pooled_gain(u, win=5)
        # Interior cells within the 5x5 window of the spike see 25/25 = 1.0
        assert g[5, 5] == pytest.approx(1.0)
        assert g[3, 3] == pytest.approx(1.0)
        assert g[2, 2] == pytest.approx(0.0)  # outside the window
        # Uniform field stays uniform (edge counts handled correctly)
        g2 = pooled_gain(np.ones((11, 11), dtype=np.float32), win=5)
        np.testing.assert_allclose(g2, 1.0, atol=1e-5)

    def test_epistemic_beta_affect_modulation(self):
        base = epistemic_beta(0.3)
        assert epistemic_beta(0.3, arousal=1.0) > base          # curiosity
        assert epistemic_beta(0.3, pain=1.0) < base             # fear
        assert epistemic_beta(0.3, pain=1.0, k_fear=1.0) == 0.0
        assert epistemic_beta(0.3, pain=1.0, k_fear=2.0) == 0.0  # clamped


class NeverDoneForage(ForageWorld):
    """ForageWorld that only terminates at the step cap (coverage probes)."""

    def step(self, a):
        obs, r, done, info = super().step(a)
        return obs, r, (self.t >= self.max_steps), info


def coverage_run(seed, use_infogain, episodes=1, H=25, W=25, steps=300):
    env = NeverDoneForage(EnvConfig(H=H, W=W, p_wall=0.15, n_targets_A=0,
                                    n_targets_B=0, max_steps=steps, seed=seed))
    env.reset()
    cfg = AgentConfig(seed=seed, epistemic_mode=("infogain" if use_infogain else "none"))
    agent = FieldController(env, ForageAdapter(env), cfg, Ablations(schema=0), seed=seed)
    covs = []
    for _ in range(episodes):
        _, _, m, _ = run_episode(env, agent, None, Ablations(schema=0))
        covs.append(m.coverage)
    return float(np.mean(covs))


class TestExploration:
    def test_epistemic_reward_vanishes_after_full_observation(self):
        """Sealed, fully-seen, fully-disconfirmed room: the epistemic term
        contributes ~nothing to the reward injection."""
        env = ForageWorld(EnvConfig(H=9, W=9, n_targets_A=0, n_targets_B=0, seed=0))
        env.reset()
        cfg = AgentConfig(seed=0)
        agent = FieldController(env, ForageAdapter(env), cfg, Ablations(), seed=0)
        agent.seen[:] = True
        agent.L["A"][:] = -8.0
        agent.L["B"][:] = -8.0
        from efi.core.infogain import pooled_gain, uncertainty_map
        u = uncertainty_map(agent.L["A"], agent.L["B"], agent.seen, agent.known_walls)
        r_epist = cfg.beta_epist * pooled_gain(u, agent.win)
        assert float(r_epist.max()) < 0.01

    def test_coverage_gate_infogain_vs_no_epistemic_term(self):
        """25x25 maze, 300 steps: infogain exploration beats the
        no-epistemic-term ablation consistently. (The ablation is not inert:
        the belief-prior optimism floor plus trail costs already produce
        implicit exploration; the epistemic term must still add real
        coverage on top, on nearly every map.)"""
        on = [coverage_run(s, True) for s in range(5)]
        off = [coverage_run(s, False) for s in range(5)]
        wins = sum(1 for a, b in zip(on, off) if a > b)
        assert wins >= 4
        assert np.mean(on) > np.mean(off) + 0.05

    def test_coverage_gate_vs_legacy_gadgets(self):
        """Infogain (lmdp) >= the legacy Novel+Frontier controller (legacy
        composition path), same maps and budget. Acceptance: not worse;
        expected: better."""
        def legacy_coverage(seed):
            env = NeverDoneForage(EnvConfig(H=25, W=25, p_wall=0.15, n_targets_A=0,
                                            n_targets_B=0, max_steps=300, seed=seed))
            env.reset()
            cfg = AgentConfig(seed=seed, control_mode="legacy", use_belief_fields=False)
            agent = FieldController(env, ForageAdapter(env), cfg,
                                    Ablations(schema=0), seed=seed)
            _, _, m, _ = run_episode(env, agent, None, Ablations(schema=0))
            return m.coverage

        seeds = range(8)
        cov_new = np.mean([coverage_run(s, True) for s in seeds])
        cov_old = np.mean([legacy_coverage(s) for s in seeds])
        assert cov_new >= cov_old - 0.02  # not worse (tolerance for seed noise)


class TestIntegration:
    def test_full_eval_not_worse_than_frontier_fallback(self):
        """Return with infogain within noise of (or better than) the
        frontier-injection fallback on the standard task."""
        from efi.evaluation import run_experiment
        env_cfg = EnvConfig(H=15, W=15, max_steps=200)
        r_on = run_experiment(env_cfg, AgentConfig(valA_init=1.0, epistemic_mode='infogain'),
                              None, Ablations(schema=0), episodes=15, seeds=2,
                              use_controller=True)
        # Like-for-like: same beliefs, same planner; only the epistemic
        # term differs (infogain vs legacy frontier attractor).
        cfg_frontier = AgentConfig(valA_init=1.0, epistemic_mode='frontier')
        r_frontier = run_experiment(env_cfg, cfg_frontier, None, Ablations(schema=0),
                                    episodes=15, seeds=2, use_controller=True)
        assert r_on.mean_return >= r_frontier.mean_return - 0.5
