"""Empirical support, recognition, and locality contracts for EFI-02."""

from dataclasses import replace

import numpy as np

from efi.agents.context_memory import ContextMemory, ContextMemoryConfig, ContextFieldController
from efi.core.experience import Experience
from efi.core.anticipation import MOTIONS
from efi.envs.context_world import ContextWorld, ContextWorldConfig
from efi.envs.command_contact import CommandContactWorld


def attempt(schema, effect, *, partial=False, learn=True):
    p = schema.table()[0, 2].astype(float)
    support = np.zeros(25)
    support[[20, 22, 23, 24]] = 1
    p *= support
    p /= p.sum()
    exp = Experience(schema.observed, 0, 2, 0, tuple(p), schema.version, (0, 0), (-1, 0))
    schema.capture(exp)
    after = None if partial else tuple(np.asarray((-1, 0)) + MOTIONS[effect])
    result = schema.update(exp, (0, 0), after, learn)
    return exp, result


def test_first_fitted_sample_is_not_archive_validation():
    model = ContextMemory(ContextMemoryConfig(validation_window=2))
    attempt(model, 2)
    assert model.admissions == 0
    attempt(model, 2)
    assert model.admissions == 0
    attempt(model, 2)
    assert model.admissions == 1
    assert model.last_event["observation"] == 3


def test_recognition_reuses_a_retained_model_after_real_feedback():
    model = ContextMemory()
    model.active[:2] = True
    model.archives[0, 0, 2, 22] = 30
    model.archives[1, 0, 2, 23] = 30
    model.archives[0, 0, 3, 23] = 30
    model.archives[1, 0, 3, 22] = 30
    model.counts[0, 2, 22] = 2
    model.weights[:] = [0.05, 0.94, 0.01, 0]
    old = model.archives[0].copy()
    for _ in range(2):
        attempt(model, 3)
    assert model.weights[2] > 0.9
    assert np.max(np.abs(model.archives[0] - old)) < 0.001
    assert np.argmax(model.table()[0, 2]) == 23
    assert model.observed == 2
    # Feedback for command 2 also changes the prediction for command 3,
    # without collecting new command-3 evidence: applicability links rows.
    assert np.argmax(model.table()[0, 3]) == 22
    assert model.counts[0, 3].sum() == 0


def test_missing_feedback_and_frozen_calls_cannot_create_empirical_support():
    model = ContextMemory()
    for _ in range(12):
        attempt(model, 2)
    before = (
        model.counts.copy(),
        model.archives.copy(),
        model.observed,
        list(model.validation),
        model.admissions,
    )
    for _ in range(20):
        attempt(model, 3, partial=True)
    assert np.array_equal(before[0], model.counts)
    assert np.array_equal(before[1], model.archives)
    assert before[2:] == (model.observed, list(model.validation), model.admissions)
    weights = model.weights.copy()
    attempt(model, 3, learn=False)
    assert np.array_equal(before[0], model.counts)
    assert np.array_equal(before[1], model.archives)
    assert np.array_equal(weights, model.weights)


def test_evidence_snapshots_are_bounded_and_eviction_is_reported():
    model = ContextMemory(ContextMemoryConfig(capacity=1, validation_window=2))
    storage = model.storage_bytes
    for effect in (2, 3, 0, 2):
        for _ in range(25):
            attempt(model, effect)
    assert model.admissions >= 3 and model.evictions == model.admissions - 1
    assert model.active.sum() == 1 and model.storage_bytes == storage
    assert len(model.recent) <= 32 and len(model.validation) <= 2


def test_prediction_capture_is_immutable_and_matches_constrained_mixture():
    model = ContextMemory()
    model.active[0] = True
    model.weights[:2] = [0.2, 0.8]
    model.archives[0, 0, 2, 22] = 10
    exp, _ = attempt(model, 2, learn=False)
    model.capture(exp)
    _, p, weights = model.saved
    np.testing.assert_allclose(weights @ p, exp.probabilities, atol=1e-7)
    snapshot = p.copy()
    model.archives.fill(99)
    assert np.array_equal(snapshot, p)


def test_imagination_and_distant_memory_cannot_change_local_evidence_or_policy():
    env = CommandContactWorld()
    obs = env.reset()
    agents = [ContextFieldController(seed=71000) for _ in range(2)]
    for agent in agents:
        agent.schema.active[0] = True
        agent.schema.archives[0, 4, 3, 23] = 10
        agent.schema.weights[:2] = [0.1, 0.9]
    agents[1].memory[2, 2, 1] = 100
    agents[1].rules.versions[2, 2] = 999
    agents[1].rules.values[2, 2] = 1 / 25
    for agent in agents:
        before = agent.schema.archives.copy()
        agent.observe(obs)
        for _ in range(3):
            agent.think()
        assert np.array_equal(before, agent.schema.archives)
        assert agent.schema.observed == 0
    np.testing.assert_allclose(agents[0].action_values, agents[1].action_values)
    for agent in agents:
        agent.select_action(forced=2)
    observation, _, _, info = env.step(2)
    for agent in agents:
        agent.after_env_step(info["displacement"])
        agent.observe(observation)
        agent.think()
        assert agent.schema.observed == 1
    for field in ("counts", "archives", "weights"):
        np.testing.assert_array_equal(
            getattr(agents[0].schema, field), getattr(agents[1].schema, field)
        )
    np.testing.assert_allclose(agents[0].action_values, agents[1].action_values)


def test_hidden_condition_changes_do_not_reset_body_or_add_observation_labels():
    cfg = ContextWorldConfig(seed=71001, min_segment=16, max_segment=16)
    a, b = ContextWorld(cfg), ContextWorld(replace(cfg, delay=3))
    np.testing.assert_array_equal(a.reset(), b.reset())
    for _ in range(16):
        obs, _, _, _ = a.step(4)
    body, block = a.body, a.occupant
    new_obs, _, _, info = a.step(4)
    assert info["phase"] == 1 and a.body == body and a.occupant == block
    np.testing.assert_array_equal(obs, new_obs)


def test_shared_record_replay_matches_actual_field_predictions():
    from efi.evaluation.context_memory import common_stream
    from efi.evaluation.context_replay import replay

    stream = common_stream(71002, modes=("bank", "fast", "no_reuse", "no_history"))
    original = {r["tick"]: r for r in stream["rows"]}
    for mode in ("bank", "fast", "no_reuse", "no_history"):
        rows, schema = replay(stream, mode)
        for row in rows:
            expected = original[row["tick"]]["scores"][mode]
            assert row["wrong"] == expected["wrong"]
            assert row["joint_loss"] == expected["joint_loss"]
        assert schema.observed == stream["final_memory"][mode]["observed"]
        if mode != "fast":
            assert schema.admissions == stream["final_memory"][mode]["admissions"]
    # Disconnecting publication must not change the memory acquisition algorithm.
    for row in stream["rows"]:
        assert row["memory"]["bank"] == row["memory"]["no_reuse"]


def test_acquired_archive_contents_change_useful_action_in_fixed_scene():
    from efi.evaluation.context_diagnostics import archive_intervention

    result = archive_intervention()
    assert result["source_steps"] == 80
    for row in result["rows"]:
        assert row["intact"][1] > row["intact"][0]
        assert row["swapped"][0] > row["swapped"][1]
        np.testing.assert_allclose(row["erased"][0], row["erased"][1], atol=1e-7)
