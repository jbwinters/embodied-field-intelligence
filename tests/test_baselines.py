"""Tests for baseline agents (Task 9)."""

from dataclasses import asdict

import numpy as np
import pytest

from efi.configs import EnvConfig
from efi.envs import ForageWorld
from efi.agents.baselines import (
    AStarOracle,
    GreedyVisibleAgent,
    RandomAgent,
    TabularQ,
    make_baseline,
    run_baseline_episode,
    train_tabular_q,
)


def eval_agent(agent, env_cfg, episodes=25, seeds=2, base_seed=0):
    rows = []
    for s in range(seeds):
        env = ForageWorld(EnvConfig(**{**asdict(env_cfg), "seed": base_seed + s}))
        for _ in range(episodes):
            rows.append(run_baseline_episode(env, agent))
    return rows


class TestOrdering:
    def test_sanity_ordering_astar_greedy_random(self):
        """Ceiling > strawman > floor, with margin, on fixed seeds."""
        env_cfg = EnvConfig(H=15, W=15, p_wall=0.08, max_steps=150)
        r_astar = eval_agent(AStarOracle(seed=0), env_cfg)
        r_greedy = eval_agent(GreedyVisibleAgent(seed=0), env_cfg)
        r_random = eval_agent(RandomAgent(seed=0), env_cfg)
        m_astar = np.mean([r["return"] for r in r_astar])
        m_greedy = np.mean([r["return"] for r in r_greedy])
        m_random = np.mean([r["return"] for r in r_random])
        assert m_astar > m_greedy + 0.1
        assert m_greedy > m_random + 0.1

    def test_astar_collects_all_A_on_open_maps(self):
        """On low-wall maps the oracle's A-collection rate is ~perfect."""
        env_cfg = EnvConfig(H=13, W=13, p_wall=0.05, max_steps=200)
        rows = eval_agent(AStarOracle(seed=1), env_cfg, episodes=20, seeds=2)
        assert np.mean([r["success"] for r in rows]) >= 0.95


class TestDeterminism:
    @pytest.mark.parametrize("name", ["greedy", "astar"])
    def test_same_seed_same_trajectory(self, name):
        def trajectory(seed):
            env = ForageWorld(EnvConfig(H=13, W=13, seed=7))
            agent = make_baseline(name, seed=seed)
            obs = env.reset()
            agent.reset()
            actions = []
            for _ in range(80):
                a = agent.act(obs, env=env if agent.needs_env else None)
                actions.append(a)
                obs, _, done, _ = env.step(a)
                if done:
                    break
            return actions

        assert trajectory(3) == trajectory(3)


class TestTabularQ:
    def test_training_improves_return(self):
        """Final training window beats the first (learning happens).

        Trains on a FIXED world so window-states repeat -- the machinery
        test. (Across freshly sampled worlds, tabular Q on raw windows
        needs orders of magnitude more episodes; that sample-inefficiency
        vs EFI's zero training is the experimental story, not a unit test.)
        """
        base = EnvConfig(H=9, W=9, p_wall=0.05, n_targets_A=1, n_targets_B=0,
                         max_steps=60, seed=50_000)

        def env_factory(ep):
            return ForageWorld(base)  # same world every episode

        agent = TabularQ(seed=0, eps_start=0.3)
        curve = train_tabular_q(env_factory, agent, episodes=600, curve_window=100)
        assert curve[-1] > curve[0]
        assert agent.learning is False  # frozen after training
        assert agent.eps == agent.eps_final

    def test_frozen_agent_does_not_update(self):
        agent = TabularQ(seed=1)
        agent.freeze()
        obs = np.zeros(100, dtype=np.float32)
        obs2 = np.ones(100, dtype=np.float32)
        agent.observe(obs, 0, 1.0, obs2, False)
        assert len(agent.Q) == 0


class TestGreedyVisible:
    def test_moves_toward_visible_A(self):
        """A directly above the agent -> greedy picks 'up'."""
        win = 5
        patch = np.zeros((4, win, win), dtype=np.float32)
        patch[1, 0, 2] = 1.0  # A two cells up
        agent = GreedyVisibleAgent(seed=0, win=win)
        assert agent.act(patch.reshape(-1)) == 0

    def test_never_steps_onto_visible_wall(self):
        win = 5
        patch = np.zeros((4, win, win), dtype=np.float32)
        patch[1, 0, 2] = 1.0   # A up
        patch[0, 1, 2] = 1.0   # wall directly up
        agent = GreedyVisibleAgent(seed=0, win=win)
        for _ in range(30):
            assert agent.act(patch.reshape(-1)) != 0

    def test_avoids_visible_B_when_alternative_exists(self):
        win = 5
        patch = np.zeros((4, win, win), dtype=np.float32)
        patch[2, 1, 2] = 1.0   # B directly up
        agent = GreedyVisibleAgent(seed=0, win=win)
        for _ in range(30):
            assert agent.act(patch.reshape(-1)) != 0
