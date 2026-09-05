"""Causal and geometric checks of reuse across motion tasks."""

import numpy as np
import pytest

from efi.agents.motion_schema import MotionSchema
from efi.agents.relational_motion import RelationalMotionSchema
from efi.core.anticipation import arrival_values, action_probabilities
from efi.envs.interception_world import InterceptionConfig, InterceptionWorld
from efi.evaluation.transfer import (
    make_transfer_agent,
    run_interception_episode,
    transfer_experiment,
)


def teach_corridor(model):
    walls = np.ones(model.shape, dtype=bool)
    walls[1:-1, 5] = False
    for _ in range(8):
        model.reset()
        for y in range(2, 8):
            occupied = np.zeros(model.shape, dtype=bool)
            occupied[y, 5] = True
            model.observe(occupied, np.ones_like(walls), walls)


def test_new_geometry_and_heading_reuse_observed_motion():
    learned = RelationalMotionSchema((11, 11))
    teach_corridor(learned)
    exact = MotionSchema(learned.shape)
    exact.counts[:] = learned.counts
    # Training was downward in a narrow vertical corridor. Test rightward
    # motion in an open room; neither heading nor wall context was trained.
    open_walls = np.zeros(learned.shape, dtype=bool)
    assert learned.kernels(open_walls)[3, 5, 5, 3] > 0.95
    assert exact.kernels(open_walls)[3, 5, 5, 3] == pytest.approx(0.2)


def test_relational_prior_does_not_supply_momentum():
    model = RelationalMotionSchema((7, 7))
    assert np.allclose(model.kernels(np.zeros(model.shape, bool))[:, 3, 3], 0.2)
    # Reversal evidence is shared just as readily as continuation evidence.
    model.counts[0, 12, 1] = 8
    assert model.kernels(np.zeros(model.shape, bool))[3, 3, 3, 2] > 0.95


def test_specific_context_evidence_overrides_general_backoff():
    model = RelationalMotionSchema((7, 7))
    model.counts[0, 12, 0] = 10  # continue in corridors
    model.counts[0, 0, 1] = 10  # reverse in open space
    kernels = model.kernels(np.zeros(model.shape, bool))
    assert kernels[3, 3, 3, 2] > 0.94


def test_relational_forecasts_rotate_and_respect_the_light_cone():
    model = RelationalMotionSchema((11, 11))
    teach_corridor(model)
    walls = np.zeros(model.shape, dtype=bool)
    walls[2, 3:8] = True
    model.mass[:] = 0
    model.mass[0, 5, 5] = 1
    predicted = model.forecast(walls, 3)
    model.mass[:] = 0
    model.mass[2, 5, 5] = 1
    rotated = model.forecast(np.rot90(walls), 3)
    yy, xx = np.indices(model.shape)
    for h, (a, b) in enumerate(zip(predicted, rotated), 1):
        assert np.allclose(np.rot90(a), b, atol=1e-7)
        assert not a[np.abs(yy - 5) + np.abs(xx - 5) > h].any()
        assert a.sum() == pytest.approx(1, abs=1e-6)


def test_reward_is_collected_once_and_future_interception_changes_action():
    walls = np.ones((3, 5), bool)
    walls[1] = False
    zero = np.zeros(walls.shape, np.float32)
    costs = np.full(walls.shape, 0.01, np.float32)
    goal = zero.copy()
    goal[1, 2] = 1
    # Certain collection on the next step terminates; distant, arbitrarily
    # large continuation must not be credited as a second reward.
    scores = arrival_values(zero + 10, costs, [zero] * 4, walls, 0.02, 2, targets=[goal] * 4)
    assert scores[3, 1, 1] == pytest.approx(0.99, abs=1e-6)
    near = zero.copy()
    near[1, 1] = 1
    # From x=2, current x=1 target will move through x=2 to x=3. Waiting
    # intercepts now; pursuing the current location misses the first arrival.
    learned = arrival_values(
        zero, costs, [zero] * 2, walls, 0.02, 2, targets=[goal, np.roll(goal, 1, axis=1)]
    )
    static = arrival_values(zero, costs, [zero] * 2, walls, 0.02, 2, targets=[near] * 2)
    assert action_probabilities(learned[:, 1, 2], 0.02)[4] > 0.6
    assert action_probabilities(static[:, 1, 2], 0.02)[2] > 0.6


def test_target_observations_share_rules_but_not_spatial_memory():
    agent = make_transfer_agent(mode="online")
    env = InterceptionWorld(InterceptionConfig(seed=4))
    obs = env.reset()
    agent.reset()
    agent.observe(obs)
    assert agent.target_motion.counts is agent.motion.counts
    assert not agent.motion.mass.any()
    assert agent.target_motion.mass.sum() == pytest.approx(1)
    assert not np.shares_memory(agent.target_motion.mass, agent.motion.mass)
    agent.target_motion.counts[0, 0, 0] = 5
    agent.reset()
    assert agent.motion.counts[0, 0, 0] == 5
    assert not agent.target_motion.mass.any()


def test_frozen_transfer_uses_no_target_learning():
    agent = make_transfer_agent()
    teach_corridor(agent.motion)
    before = agent.motion.counts.copy()
    row, _ = run_interception_episode(InterceptionWorld(InterceptionConfig(seed=8)), agent)
    assert row["learned_transitions"] == 0
    assert np.array_equal(agent.motion.counts, before)


def test_target_collection_is_synchronous_and_bumps_are_distinct_from_waits():
    a = InterceptionWorld(InterceptionConfig(seed=3, obstacles=True))
    b = InterceptionWorld(InterceptionConfig(seed=3, obstacles=True))
    assert np.array_equal(a.reset(), b.reset())
    for action in (0, 4, 2, 3):
        left, right = a.step(action), b.step(action)
        assert np.array_equal(left[0], right[0])
        assert left[1:] == right[1:]
    a.target_index = 4
    a.direction = 1
    a.y, a.x = a.track[5]
    _, reward, done, info = a.step(4)
    assert done and info["success"] and reward == pytest.approx(0.99)
    assert info["wait"] and not info["bump"]
    assert not a.observation().reshape(6, a.win, a.win)[5].any()


def test_zero_target_fields_preserve_existing_action_values():
    rng = np.random.RandomState(42)
    terminal = rng.uniform(size=(7, 7)).astype(np.float32)
    costs = np.full_like(terminal, 0.01)
    hazard = np.zeros_like(terminal)
    walls = np.zeros_like(terminal, dtype=bool)
    original = arrival_values(terminal, costs, [hazard] * 4, walls, 0.02, 2)
    extended = arrival_values(terminal, costs, [hazard] * 4, walls, 0.02, 2, targets=[hazard] * 4)
    assert np.array_equal(original, extended)


def test_correction_wave_removes_stale_hypotheses_only_within_its_budget():
    model = RelationalMotionSchema((25, 25))
    walls = np.zeros(model.shape, bool)
    visible = np.zeros_like(walls)
    visible[11:14, 11:14] = True
    occupied = np.zeros_like(walls)
    occupied[12, 12] = True
    model.mass[4, 12, 16] = model.mass[4, 12, 22] = 1
    model.trace_time[12, 16] = model.trace_time[12, 22] = 0
    model.observe(occupied, visible, walls, learn=False)
    assert not model.mass[:, 12, 16].any()
    assert model.mass[:, 12, 22].sum() > 0
    assert model.mass[:, 12, 12].sum() == pytest.approx(1)
    yy, xx = np.indices(model.shape)
    distance = np.abs(yy - 12) + np.abs(xx - 12)
    assert np.all(model.evidence_time[distance > model.correction_sweeps] == -1)


def test_occluded_object_persists_without_newer_contradictory_evidence():
    model = RelationalMotionSchema((15, 15))
    occupied = np.zeros(model.shape, bool)
    occupied[7, 7] = True
    walls = np.zeros_like(occupied)
    model.observe(occupied, np.ones_like(occupied), walls)
    for _ in range(3):
        model.observe(np.zeros_like(occupied), np.zeros_like(occupied), walls)
        assert model.mass.sum() == pytest.approx(1, abs=1e-6)


def test_unseen_geometry_is_not_treated_as_certain_free_space():
    model = RelationalMotionSchema((11, 11))
    teach_corridor(model)
    model.observed_space[:] = True
    walls = np.zeros(model.shape, bool)
    certain = model.kernels(walls)[3, 5, 5, 3]
    model.observed_space[5, 6] = False
    uncertain = model.kernels(walls)[3, 5, 5, 3]
    assert certain > 0.95
    assert uncertain < 0.6
    assert model.kernels(walls)[3, 5, 5].sum() == pytest.approx(1)


def test_all_motion_channels_are_equivariant_including_stationary_objects():
    model = RelationalMotionSchema((9, 9))
    rng = np.random.RandomState(17)
    model.counts[:] = rng.uniform(size=model.counts.shape)
    model.mass[:] = rng.uniform(size=model.mass.shape)
    walls = np.zeros(model.shape, bool)
    walls[3:5, 2] = True
    model.mass[:, walls] = 0
    original_mass = model.mass.copy()
    forecast = model.forecast(walls, 2)
    for d, rotated in enumerate((2, 3, 1, 0, 4)):
        model.mass[rotated] = np.rot90(original_mass[d])
    rotated = model.forecast(np.rot90(walls), 2)
    for a, b in zip(forecast, rotated):
        assert np.allclose(np.rot90(a), b, atol=1e-6)


def test_relational_tracking_explicitly_rejects_multiple_objects_per_channel():
    model = RelationalMotionSchema((7, 7))
    occupied = np.zeros(model.shape, bool)
    occupied[3, 2] = occupied[3, 4] = True
    with pytest.raises(ValueError, match="one isolated object"):
        model.observe(occupied, np.ones_like(occupied), np.zeros_like(occupied))


def test_interception_collision_takes_precedence_over_reward_and_detects_swaps():
    env = InterceptionWorld(InterceptionConfig(seed=3, hazards=True))
    env.reset()
    crossing = next(point for point in env.track if point in env.hazard_track)
    env.target_index = env.track.index(crossing) - 1
    env.direction = 1
    env.hazard_index = env.hazard_track.index(crossing) - 1
    env.hazard_direction = 1
    env.y, env.x = crossing
    _, reward, done, info = env.step(4)
    assert done and info["collision"] and not info["success"]
    assert reward == pytest.approx(-2.01)
    env.hazard_index = env.hazard_track.index(crossing)
    env.hazard_direction = 1
    env.y, env.x = env.hazard_track[env.hazard_index + 1]
    _, _, done, info = env.step(0)
    assert done and info["collision"]


def test_cross_task_reuse_on_separate_behavioral_test_seeds():
    result = transfer_experiment(seeds=4, episodes=6, acquisition=20, base_seed=7000, record=False)
    for task in ("obstacles", "mixed_obstacles"):
        summary = result["summary"][task]
        assert summary["transfer"]["success"] >= 0.65
        assert summary["transfer"]["success"] > summary["exact"]["success"] + 0.15
        assert summary["transfer"]["return"] > summary["static"]["return"] + 0.1
    assert all(
        r["learned_transitions"] == 0
        for r in result["rows"]
        if r["mode"] not in ("online", "scratch_online")
    )
