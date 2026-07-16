"""Tests for the egocentric controller (Task 7): no GPS, no world size."""

import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from efi.configs import AgentConfig, Ablations, EnvConfig
from efi.envs import ForageWorld
from efi.agents import EgocentricFieldController, FieldController, ForageAdapter
from efi.evaluation import run_ego_episode, run_episode

CONTROLLER_SRC = (Path(__file__).parent.parent
                  / "efi" / "agents" / "egocentric_controller.py")


def make_ego(seed=0, **cfg_kwargs):
    cfg = AgentConfig(valA_init=1.0, seed=seed, **cfg_kwargs)
    return EgocentricFieldController(cfg, Ablations(schema=0), win=5, seed=seed)


class TestNoForbiddenReads:
    def test_source_never_touches_env_truth(self):
        """The whole point: AST-check the controller CODE (not docstrings)
        for any attribute access on an `env` object."""
        import ast

        tree = ast.parse(CONTROLLER_SRC.read_text())
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                v = node.value
                if (isinstance(v, ast.Name) and v.id == "env") or (
                        isinstance(v, ast.Attribute) and v.attr == "env"):
                    offenders.append((node.lineno, node.attr))
        assert not offenders, f"egocentric controller reads env truth: {offenders}"

    def test_constructor_takes_no_env(self):
        import inspect
        params = inspect.signature(EgocentricFieldController.__init__).parameters
        assert "env" not in params


class TestPerfectOdometry:
    def test_performance_parity_with_world_frame_controller(self):
        """With perfect odometry, removing the GPS should not cost much.
        Behavior legitimately differs where the world-frame controller was
        cheating (it knows world bounds a priori; the ego agent must
        discover them), so parity is measured on outcomes, not actions."""
        n = 12
        returns_world, returns_ego = [], []
        succ_world = succ_ego = 0
        for seed in range(n):
            env_cfg = EnvConfig(H=15, W=15, max_steps=200, seed=seed)
            env = ForageWorld(env_cfg)
            env.reset()
            cfg = AgentConfig(valA_init=1.0, seed=seed)
            agent_w = FieldController(env, ForageAdapter(env), cfg,
                                      Ablations(schema=0), seed=seed)
            ret_w, _, m_w, _ = run_episode(env, agent_w, None, Ablations(schema=0))
            returns_world.append(ret_w)
            succ_world += m_w.targets_collected.get("A", 0) >= env_cfg.n_targets_A

            env2 = ForageWorld(env_cfg)
            agent_e = make_ego(seed=seed)
            ret_e, m_e = run_ego_episode(env2, agent_e)
            returns_ego.append(ret_e)
            succ_ego += m_e.targets_collected.get("A", 0) >= env_cfg.n_targets_A
            # Perfect odometry: dead reckoning must be exact
            assert m_e.final_pose_error == 0.0

        assert succ_ego >= succ_world - 2  # allow 2 episodes of slack
        assert np.mean(returns_ego) >= np.mean(returns_world) - 0.4

    def test_pose_tracks_displacement_exactly(self):
        env = ForageWorld(EnvConfig(H=13, W=13, seed=3))
        env.reset()
        agent = make_ego(seed=3)
        agent.reset()
        p0, e0 = agent.pose, (env.y, env.x)
        for _ in range(60):
            agent.observe(env._obs())
            agent.think(None)
            a = agent.select_action()
            _, _, done, info = env.step(a)
            agent.after_env_step(a, bool(info["moved"]), info.get("picked"))
            dp = (agent.pose[0] - p0[0], agent.pose[1] - p0[1])
            de = (env.y - e0[0], env.x - e0[1])
            assert dp == de
            if done:
                break

    def test_discovers_world_boundary_by_bumping_or_seeing(self):
        """Out-of-bounds reads as wall in observations; the internal map
        should accumulate boundary cells the agent has approached."""
        env = ForageWorld(EnvConfig(H=9, W=9, p_wall=0.0, n_targets_A=0,
                                    n_targets_B=0, max_steps=120, seed=1))
        env.reset()
        agent = make_ego(seed=1)
        run_ego_episode(env, agent)
        # Some boundary must have been mapped as wall (out-of-bounds cells)
        assert agent.known_walls.sum() > 0

    def test_map_edge_walk_does_not_crash(self):
        """Long drifting walk on a tiny internal map: defined behavior."""
        env = ForageWorld(EnvConfig(H=25, W=25, p_wall=0.0, n_targets_A=0,
                                    n_targets_B=0, max_steps=300, seed=2))
        env.reset()
        agent = make_ego(seed=2, map_size=31)  # tight map, pose nears edges
        ret, m = run_ego_episode(env, agent)
        assert np.isfinite(ret)


class StepCapForage(ForageWorld):
    """Terminates only at the step cap (a target-free world would otherwise
    end at t=1, before any odometry slip can occur)."""

    def step(self, a):
        obs, r, done, info = super().step(a)
        return obs, r, (self.t >= self.max_steps), info


class TestNoisyOdometry:
    def test_correction_beats_dead_reckoning_under_slip(self):
        """p_slip=0.1: visual odometry (consecutive-window overlap
        alignment) detects slips the instant they happen, and anchor-based
        template matching mops up residual drift. Together they must beat
        raw dead reckoning by a wide margin (measured: ~13x)."""
        errs_on, errs_off = [], []
        for seed in range(12):
            for correction, errs in [(True, errs_on), (False, errs_off)]:
                env = StepCapForage(EnvConfig(H=15, W=15, p_wall=0.15,
                                              n_targets_A=0, n_targets_B=0,
                                              max_steps=200, p_slip=0.1, seed=seed))
                env.reset()
                agent = make_ego(seed=seed, pose_correction=correction)
                _, m = run_ego_episode(env, agent)
                errs.append(m.mean_pose_error)
        assert np.mean(errs_off) > 0            # slip actually causes drift
        assert np.mean(errs_on) * 2.0 <= np.mean(errs_off)   # >=2x mean reduction
        assert np.mean(errs_on) < 1.0           # near-exact pose in absolute terms

    def test_no_slip_no_drift_regardless_of_correction(self):
        env = ForageWorld(EnvConfig(H=13, W=13, seed=5))
        env.reset()
        agent = make_ego(seed=5, pose_correction=True)
        _, m = run_ego_episode(env, agent)
        assert m.final_pose_error == 0.0
