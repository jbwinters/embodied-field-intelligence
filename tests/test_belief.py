"""Tests for log-odds Bayesian belief fields (Task 1)."""

import numpy as np
import pytest

from efi.core.belief import (
    sigmoid,
    logit,
    logodds_correct,
    logodds_predict,
    belief_to_expected_reward,
)
from efi.configs import AgentConfig, Ablations, BeliefConfig, EnvConfig
from efi.envs import ForageWorld
from efi.agents import FieldController, ForageAdapter


L_PRIOR = -4.0


def make_L(H=11, W=11, value=L_PRIOR):
    return np.full((H, W), value, dtype=np.float32)


def no_walls(H=11, W=11):
    return np.zeros((H, W), dtype=bool)


class TestBeliefPrimitives:
    def test_negative_evidence_drives_probability_down(self):
        """A cell observed empty once (hard evidence) goes to p < 0.01."""
        L = make_L(value=0.0)  # start uncertain (p = 0.5)
        L = logodds_correct(L, pos_cells=[], neg_cells=[(5, 5)], l_neg=-8.0)
        assert sigmoid(L)[5, 5] < 0.01

    def test_positive_evidence_drives_probability_up(self):
        """One noiseless observation saturates: increment exceeds the clamp."""
        L = make_L()  # prior p ~ 0.018
        L = logodds_correct(L, pos_cells=[(3, 4)], neg_cells=[], l_pos=12.0)
        assert sigmoid(L)[3, 4] > 0.99

    def test_clamp_keeps_beliefs_disconfirmable(self):
        """Repeated positive evidence saturates at l_max; ONE negative
        observation must still be able to flip it below neutral."""
        L = make_L(value=0.0)
        for _ in range(100):
            L = logodds_correct(L, pos_cells=[(2, 2)], neg_cells=[],
                                l_pos=8.0, l_max=8.0)
        assert L[2, 2] == pytest.approx(8.0)
        L = logodds_correct(L, pos_cells=[], neg_cells=[(2, 2)], l_neg=-8.0)
        assert sigmoid(L)[2, 2] <= 0.5

    def test_unobserved_cells_drift_toward_prior(self):
        """With no evidence, log-odds relax toward l_prior geometrically."""
        L = make_L(value=0.0)  # uncertain, above the prior
        walls = no_walls()
        dist_prev = abs(0.0 - L_PRIOR)
        for _ in range(50):
            L = logodds_predict(L, walls, diff=0.0, decay=0.0,
                                l_prior=L_PRIOR, rho_prior=0.05)
            dist = abs(float(L[5, 5]) - L_PRIOR)
            assert dist < dist_prev  # strictly approaching the prior
            dist_prev = dist

    def test_diffusion_does_not_leak_through_walls(self):
        """Probability mass must not cross a solid wall line."""
        H, W = 11, 11
        L = make_L(H, W, value=-8.0)
        L[2, 5] = 8.0  # confident target on the top side
        walls = no_walls(H, W)
        walls[5, :] = True  # full horizontal wall
        p_below_before = sigmoid(L)[6:, :].copy()
        for _ in range(30):
            L = logodds_predict(L, walls, diff=0.2, decay=0.0,
                                l_prior=-8.0, rho_prior=0.0)
        p_below_after = sigmoid(L)[6:, :]
        np.testing.assert_allclose(p_below_after, p_below_before, atol=1e-4)
        # sanity: mass did spread on the top side
        assert sigmoid(L)[3, 5] > sigmoid(np.float32(-8.0))

    def test_expected_reward_map(self):
        L = make_L(value=0.0)
        R = belief_to_expected_reward(L, reward=2.0)
        assert R[0, 0] == pytest.approx(1.0, abs=1e-5)  # p=0.5 * 2.0

    def test_logit_sigmoid_roundtrip_float32_stable(self):
        L = np.array([[-8.0, -1.0, 0.0, 1.0, 8.0]], dtype=np.float32)
        L2 = logit(sigmoid(L))
        np.testing.assert_allclose(L2, L, atol=1e-2)


def make_controller(seed=0, use_beliefs=True, H=17, W=17):
    env_cfg = EnvConfig(H=H, W=W, seed=seed)
    env = ForageWorld(env_cfg)
    env.reset()
    cfg = AgentConfig(seed=seed, use_belief_fields=use_beliefs)
    ablate = Ablations()
    agent = FieldController(env, ForageAdapter(env), cfg, ablate, seed=seed)
    return env, agent


class TestControllerIntegration:
    def test_seen_empty_cell_is_disconfirmed(self):
        """The controller writes negative evidence for visible empty cells:
        after one step, in-window non-wall cells without a target sit near
        p=0 -- the disconfirmation scent fields never had."""
        env, agent = make_controller(seed=3)
        obs = env._obs()
        agent.step_fields(obs)
        y, x = env.y, env.x
        half = env.win // 2
        pA = agent.fields["A"]
        prior_p = float(sigmoid(np.float32(agent.belief_cfg.l_prior)))
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                gy, gx = y + dy, x + dx
                if not (0 <= gy < env.H and 0 <= gx < env.W):
                    continue
                if env.walls[gy, gx]:
                    continue
                if env.TA[gy, gx]:
                    assert pA[gy, gx] > 0.9
                else:
                    assert pA[gy, gx] < prior_p  # pushed below the prior

    def test_belief_persists_and_rediscovery_after_disappearance(self):
        """Belief at a seen target survives leaving; re-observing the cell
        empty (after pickup elsewhere in the API) drives it back down."""
        env, agent = make_controller(seed=3)
        obs = env._obs()
        agent.step_fields(obs)
        # Find a visible A target if any; otherwise inject evidence manually
        ys, xs = np.where(env.TA)
        ty, tx = int(ys[0]), int(xs[0])
        agent.L["A"][ty, tx] = agent.belief_cfg.l_max
        agent.fields["A"][ty, tx] = 1.0
        # Remove the target from the world; belief should stay high until
        # the cell is observed again (memory), ...
        env.TA[ty, tx] = False
        assert agent.fields["A"][ty, tx] > 0.9
        # ... then fall on direct re-observation of the (now empty) cell.
        env.y, env.x = ty, tx  # teleport agent onto the cell for observation
        agent.step_fields(env._obs())
        assert agent.fields["A"][ty, tx] < 0.05

    def test_notify_pickup_zeroes_belief(self):
        env, agent = make_controller(seed=0)
        agent.L["A"][env.y, env.x] = agent.belief_cfg.l_max
        agent.notify_pickup("A")
        assert sigmoid(agent.L["A"])[env.y, env.x] < 0.001
        assert agent.fields["A"][env.y, env.x] < 0.001

    def test_legacy_scent_path_unchanged(self):
        """use_belief_fields=False must reproduce the old max-inject seeding."""
        env, agent = make_controller(seed=3, use_beliefs=False)
        obs = env._obs()
        agent.step_fields(obs)
        assert not agent.use_beliefs
        # Legacy fields are diffused scent, bounded by seed_strength
        assert agent.fields["A"].max() <= agent.cfg.seed_strength + 1e-6

    def test_episode_smoke_with_beliefs(self):
        """Full episode through the runner with beliefs on."""
        from efi.evaluation import run_episode
        env, agent = make_controller(seed=1)
        ablate = Ablations()
        ret, _, metrics, _ = run_episode(env, agent, None, ablate)
        assert np.isfinite(ret)
        assert metrics.steps > 0
        # Probability maps stay in [0, 1]
        assert 0.0 <= agent.fields["A"].min() and agent.fields["A"].max() <= 1.0
