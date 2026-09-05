"""Capability, evidence, information-boundary and locality tests."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from efi.agents.interaction_schema import InteractionSchema, ROTATE
from efi.configs.interaction_config import InteractionConfig
from efi.core.experience import Experience, RuleField, gather, transport
from efi.core.interaction import grouped_continuation
from efi.envs.interaction_world import InteractionWorld, InteractionWorldConfig
from efi.evaluation.interaction import acquire, make_agent, run_episode
from efi.evaluation.interaction_reference import scalar_continuation


@pytest.fixture(scope="module")
def acquired():
    return {law: acquire(7001, law, 2)[0] for law in ("push", "left", "right")}


def test_complete_feedback_updates_only_its_row_and_scores_before_learning():
    schema = InteractionSchema()
    e = Experience(0, 0, 0, 0, (1 / 25,) * 25, 0, (5, 5), (4, 5))
    loss, complete = schema.update(e, (-1, 0), (4, 4))
    assert complete and loss == pytest.approx(np.log(25))
    assert schema.counts[0, 0, 2] == 1  # body forward, object left
    assert schema.counts.sum() == 1
    assert e.probabilities == (1 / 25,) * 25
    with pytest.raises(FrozenInstanceError):
        e.model_version = 2


def test_partial_feedback_never_invents_joint_support():
    schema = InteractionSchema()
    e = Experience(0, 0, 0, 0, (1 / 25,) * 25, 0, (5, 5), (4, 5))
    before = schema.counts.tobytes()
    for _ in range(100):
        loss, complete = schema.update(e, (-1, 0), None)
        assert not complete and loss == pytest.approx(np.log(5))
    assert schema.counts.tobytes() == before
    assert schema.version == schema.observed == 0


def test_imagination_preserves_empirical_counts_and_saved_experience(acquired):
    agent = make_agent(7)
    agent.schema.counts[:] = acquired["left"]
    env = InteractionWorld(InteractionWorldConfig(acquisition=True))
    agent.observe(env.reset())
    agent.think()
    agent.select_action(0)
    saved = agent.pending
    before = agent.schema.counts.tobytes()
    for _ in range(20):
        agent.think()
    assert agent.schema.counts.tobytes() == before
    assert agent.pending == saved


def test_hidden_outcomes_share_future_actions():
    q = np.array([[1, -1, -1, -1, -1], [-1, 1, -1, -1, -1]], dtype=float)
    weights = np.array([0.5, 0.5])
    same = np.zeros((2, 3), dtype=int)
    value, _ = grouped_continuation(weights, same, q, 0.01)
    assert np.all(value < 0)  # no oracle can pick the right hidden action
    distinguishable = np.array([[0, 0, 0], [1, 0, 0]])
    informed, _ = grouped_continuation(weights, distinguishable, q, 0.01)
    assert np.all(informed > 0.98)


def test_scalar_reference_matches_grouped_field_for_partial_observation():
    rng = np.random.RandomState(79)
    keys = rng.randint(0, 3, (125, 3))
    weights = rng.rand(125)
    values = rng.randn(125, 5)
    actual, groups = grouped_continuation(weights, keys, values, 0.02)
    expected, reference_groups = scalar_continuation(weights, keys, values, 0.02)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert groups == reference_groups


def test_moore_gather_reaches_corners_and_respects_its_cone():
    field = np.arange(31 * 31).reshape(31, 31)
    port, work = gather(field, (15, 15), 2)
    np.testing.assert_array_equal(port, field[13:18, 13:18])
    changed = field.copy()
    changed[:13] = -123
    changed[18:] = -123
    np.testing.assert_array_equal(gather(changed, (15, 15), 2)[0], port)
    assert work > 2 * field.size  # work and propagation depth are different
    pulse = np.zeros((7, 7))
    pulse[0, 0] = 1
    assert transport(pulse, -1, -1).sum() == 0
    with pytest.raises(ValueError):
        transport(pulse, 2, 0)


def test_rule_changes_travel_locally_and_are_immutable_snapshots():
    field = RuleField(15)
    table = np.zeros((16, 5, 25), dtype=np.float32)
    table[..., 0] = 1
    field.publish((7, 7), table, 1)
    table[..., 0] = 0
    field.spread(2)
    expected = np.zeros((15, 15), dtype=bool)
    expected[5:10, 5:10] = True
    np.testing.assert_array_equal(field.versions == 1, expected)
    assert field.values[7, 7, 0, 0, 0] == 1
    newer = np.zeros_like(table)
    newer[..., 2] = 1
    field.publish((7, 7), newer, 2)
    field.spread(1)
    assert field.versions[5, 5] == 1
    assert field.values[5, 5, 0, 0, 0] == 1
    assert field.values[6, 6, 0, 0, 2] == 1


@pytest.mark.parametrize("law", ["push", "left", "right"])
def test_acquired_effects_transfer_to_rotated_rearrangements(acquired, law):
    agent = make_agent(7013, "frozen")
    agent.schema.counts[:] = acquired[law]
    outcomes = []
    for layout in ("west", "north"):
        for rotation in range(4):
            env = InteractionWorld(
                InteractionWorldConfig(rule=law, layout=layout, rotate=rotation, size=13)
            )
            outcomes.append(run_episode(env, agent)["success"])
    assert all(outcomes)
    np.testing.assert_array_equal(agent.schema.counts, acquired[law])


def test_same_scene_learned_effects_change_the_first_action(acquired):
    policies = []
    env = InteractionWorld(InteractionWorldConfig(layout="west"))
    for law in ("push", "left"):
        agent = make_agent(7005, "frozen")
        agent.schema.counts[:] = acquired[law]
        agent.observe(env.reset())
        agent.think()
        policies.append(agent.policy)
    assert policies[0][2] > 0.99  # approach from below, then push upward
    assert policies[1][0] > 0.99  # approach from the right, then cause a left yield


def test_no_hidden_world_geometry_or_rule_enters_policy():
    agent = make_agent(31)
    world = InteractionWorld(InteractionWorldConfig(rule="push"))
    obs = world.reset()
    world.cfg = InteractionWorldConfig(rule="left")
    world.walls[1, 1] = ~world.walls[1, 1]
    np.testing.assert_array_equal(world.observation(), obs)
    agent.observe(obs)
    agent.think()
    expected = agent.policy.copy()
    agent.reset()
    agent.observe(world.observation())
    agent.think()
    np.testing.assert_array_equal(agent.policy, expected)


def test_finite_zero_boundary_and_unresolved_mass_are_conservative():
    agent = make_agent(9, "empty")
    env = InteractionWorld(InteractionWorldConfig())
    agent.observe(env.reset())
    # Unsupported geometry is inside the port, not hidden truth supplied by env.
    agent.port[:, :, 0] = -1
    q = agent.think()
    assert np.all(q <= -2)
    assert np.all(np.isfinite(q))
    assert np.max(agent.field.first[0].sum(axis=1)) == 0
    assert np.min(agent.field.first[1]) == 1
    np.testing.assert_array_equal(agent.field.value_bounds[:, 0], q)
    assert np.all(agent.field.value_bounds[:, 1] >= q)


def test_terminal_reward_cannot_be_collected_twice():
    agent = make_agent(9)
    env = InteractionWorld(InteractionWorldConfig())
    obs = env.reset().reshape(5, 5, 5)
    obs[1] = 0
    obs[1, 2, 2] = 1
    agent.observe(obs.reshape(-1))
    q = agent.think()
    assert q[4] == pytest.approx(0.99)
    assert q.max() <= 0.99 + 1e-8


def test_full_decision_ignores_memory_outside_working_cone(acquired):
    agent = make_agent(41)
    agent.schema.counts[:] = acquired["push"]
    env = InteractionWorld(InteractionWorldConfig())
    obs = env.reset()
    agent.observe(obs)
    expected = agent.think().copy()
    y, x = agent.pose
    outside = np.ones(agent.memory.shape[:2], dtype=bool)
    outside[y - 4 : y + 5, x - 4 : x + 5] = False
    agent.memory[outside] = 0.7
    agent.observe(obs)
    np.testing.assert_allclose(agent.think(), expected, atol=1e-12)


def test_capacity_and_work_remain_bounded(acquired):
    agent = make_agent(9)
    agent.schema.counts[:] = acquired["push"]
    env = InteractionWorld(InteractionWorldConfig())
    agent.observe(env.reset())
    size = agent.nbytes
    for _ in range(40):
        run_episode(env, agent)
        assert agent.nbytes == size
        assert agent.field.outcome_terms <= 16900
    assert len(agent.history) == 32
    assert size < 32 * 1024**2


def test_episode_final_transition_is_learned():
    agent = make_agent(9, source=True)
    env = InteractionWorld(InteractionWorldConfig(acquisition=True, max_steps=1))
    run_episode(env, agent, forced=0)
    assert agent.schema.observed == 1


def test_multiple_objects_rejected_and_config_budget_enforced():
    agent = make_agent()
    obs = np.zeros((5, 5, 5), dtype=np.float32)
    obs[4, 1, 2] = obs[4, 2, 1] = 1
    with pytest.raises(ValueError):
        agent.observe(obs.reshape(-1))
    with pytest.raises(ValueError):
        InteractionConfig(horizon=3)
    with pytest.raises(ValueError):
        InteractionConfig(rule_passes=3)
    with pytest.raises(ValueError):
        InteractionConfig(rule_passes=0.5)


def test_action_relative_frame_has_no_preferred_world_orientation():
    assert len(set(ROTATE[:, 0])) == 4
    np.testing.assert_array_equal(ROTATE[:, 4], 4)


def test_rendered_horizon_prediction_is_bounded_and_does_not_train(acquired):
    agent = make_agent(42)
    agent.schema.counts[:] = acquired["push"]
    env = InteractionWorld(InteractionWorldConfig())
    agent.observe(env.reset())
    agent.think()
    before = agent.schema.counts.tobytes()
    positions, mass = agent.field.object_forecast(agent.policy)
    assert len(positions) == len(mass)
    assert 0 < mass.sum() <= 1 + 1e-10
    assert agent.schema.counts.tobytes() == before


def test_unseen_rewards_remain_in_upper_bound_and_empty_scene_can_render():
    from efi.evaluation.interaction import snapshot

    agent = make_agent(43)
    env = InteractionWorld(InteractionWorldConfig(acquisition=True))
    obs = env.reset().reshape(5, 5, 5)
    obs[4] = 0
    agent.observe(obs.reshape(-1))
    agent.think()
    assert np.all(agent.field.value_bounds[:, 1] == 1.98)
    fields, world = snapshot(env, agent, 0, "No visible object")
    assert np.isfinite(fields["info"]["value_bounds"]).all()
    assert world.shape == (13, 13, 3)
