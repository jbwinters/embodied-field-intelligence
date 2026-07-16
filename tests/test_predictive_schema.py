"""Tests for the predictive schema (Task 8)."""

import numpy as np
import pytest

from efi.configs import AgentConfig, Ablations, EnvConfig
from efi.envs import ForageWorld
from efi.agents import EgocentricFieldController, FieldController, ForageAdapter
from efi.agents.predictive_schema import PredictiveSchema
from efi.evaluation import run_ego_episode, run_episode


def scripted_corridor_walk(schema, env, actions):
    """Drive the env with a fixed action sequence, feeding transitions to
    the schema; returns per-step surprise."""
    surprises = []
    prev = env._obs().reshape(4, env.win, env.win)
    for a in actions:
        _, _, _, info = env.step(a)
        cur = env._obs().reshape(4, env.win, env.win)
        s = schema.observe_transition(prev, a, bool(info["moved"]), cur)
        surprises.append(s)
        prev = cur
    return surprises


class CorridorEnv(ForageWorld):
    """Deterministic empty corridor; episode never ends early."""

    def reset(self):
        obs = super().reset()
        self.walls[:] = False
        self.TA[:] = False
        self.TB[:] = False
        self.y, self.x = self.H // 2, 2
        return self._obs()

    def step(self, a):
        obs, r, done, info = super().step(a)
        return obs, r, (self.t >= self.max_steps), info


class TestLearning:
    def test_static_cell_accuracy_after_experience(self):
        """After 200 steps in a static world, held-out per-cell prediction
        accuracy on familiar contexts exceeds 95%."""
        env = ForageWorld(EnvConfig(H=15, W=15, max_steps=200, seed=0))
        env.reset()
        cfg = AgentConfig(valA_init=1.0, seed=0)
        agent = FieldController(env, ForageAdapter(env), cfg, Ablations(schema=0), seed=0)
        run_episode(env, agent, None, Ablations(schema=0))
        assert agent.pschema is not None
        assert agent.pschema.heldout_total > 20  # enough held-out samples
        assert agent.pschema.heldout_accuracy > 0.95

    def test_surprise_decays_with_familiarity(self):
        """First pass through a corridor: everything unfamiliar (surprise
        ~1). Third pass over the same cells with the same actions: the rule
        is known (surprise near 0)."""
        env = CorridorEnv(EnvConfig(H=7, W=24, p_wall=0.0, n_targets_A=0,
                                    n_targets_B=0, max_steps=10_000, seed=1))
        env.reset()
        schema = PredictiveSchema(win=env.win)
        right, left = [3] * 18, [2] * 18
        pass1 = scripted_corridor_walk(schema, env, right)
        scripted_corridor_walk(schema, env, left)   # walk back
        pass3 = scripted_corridor_walk(schema, env, right)
        # The very first transition is maximally surprising (no rules yet);
        # a uniform corridor is then learned in ONE step (one-shot rule
        # learning) and stays calm on revisits.
        assert pass1[0] == pytest.approx(1.0)
        assert np.mean(pass3) < 0.1
        assert np.mean(pass1) > np.mean(pass3)

    def test_imagination_reproduces_memorized_corridor(self):
        """Sensor-detached rollout along a memorized corridor matches the
        true windows on every cell the schema is confident about."""
        env = CorridorEnv(EnvConfig(H=7, W=24, p_wall=0.0, n_targets_A=0,
                                    n_targets_B=0, max_steps=10_000, seed=2))
        env.reset()
        schema = PredictiveSchema(win=env.win)
        # Memorize: two passes over the corridor
        scripted_corridor_walk(schema, env, [3] * 18)
        scripted_corridor_walk(schema, env, [2] * 18)
        scripted_corridor_walk(schema, env, [3] * 9)  # stop mid-corridor

        start_patch = env._obs().reshape(4, env.win, env.win)
        imagined = schema.imagine(start_patch, [3, 3, 3])

        # Ground truth: actually walk those 3 steps
        for k, a in enumerate([3, 3, 3]):
            env.step(a)
            actual = env._obs().reshape(4, env.win, env.win)
            pred, conf = imagined[k]
            confident = conf >= 0.8
            assert confident.any()  # imagination is not vacuous
            for ch in range(3):  # walls, A, B (agent channel excluded)
                mism = confident & (np.abs(pred[ch] - actual[ch]) > 0.5)
                assert not mism.any(), f"step {k} channel {ch}"

    def test_static_confidence_rises_in_static_world(self):
        env = ForageWorld(EnvConfig(H=15, W=15, max_steps=200, seed=3))
        env.reset()
        cfg = AgentConfig(valA_init=1.0, seed=3)
        agent = FieldController(env, ForageAdapter(env), cfg, Ablations(schema=0), seed=3)
        run_episode(env, agent, None, Ablations(schema=0))
        run_episode(env, agent, None, Ablations(schema=0))
        assert agent.pschema.static_confidence > 0.8


class TestIntegration:
    def test_return_gate_predictive_vs_off(self):
        """The bar the Oja schema failed: predictive mode must not cost
        return (>= off - 0.1) on the standard eval."""
        from efi.evaluation import run_experiment
        env_cfg = EnvConfig(H=15, W=15, max_steps=200)
        rets = {}
        for mode in ("predictive", "off"):
            cfg = AgentConfig(valA_init=1.0, schema_mode=mode)
            r = run_experiment(env_cfg, cfg, None, Ablations(schema=0),
                               episodes=20, seeds=3, use_controller=True)
            rets[mode] = r.mean_return
        assert rets["predictive"] >= rets["off"] - 0.1

    def test_ego_controller_carries_schema(self):
        env = ForageWorld(EnvConfig(H=13, W=13, max_steps=150, seed=4))
        env.reset()
        cfg = AgentConfig(valA_init=1.0, seed=4)
        agent = EgocentricFieldController(cfg, Ablations(schema=0), win=5, seed=4)
        _, m = run_ego_episode(env, agent)
        assert agent.pschema is not None
        assert m.schema_rules > 0
        assert agent.pschema._updates > 50

    def test_metrics_report_heldout_accuracy(self):
        env = ForageWorld(EnvConfig(H=13, W=13, max_steps=150, seed=5))
        env.reset()
        cfg = AgentConfig(valA_init=1.0, seed=5)
        agent = FieldController(env, ForageAdapter(env), cfg, Ablations(schema=0), seed=5)
        _, _, m, _ = run_episode(env, agent, None, Ablations(schema=0))
        assert 0.0 <= m.accuracy_predictive <= 1.0
        assert m.schema_rules > 0
