"""Behavioral checks for EFI-01's command-specific evidence contract."""

import numpy as np
import pytest

from efi.envs.command_contact import CommandContactConfig, CommandContactWorld
from efi.evaluation.command_contact import (
    acquire,
    make_agent,
    probe,
    rotated_command,
    run_target,
)


@pytest.mark.parametrize("rotation", range(4))
def test_commands_change_object_without_body_or_initial_sensation_cue(rotation):
    observations, effects = [], {}
    for law in ("aligned", "reversed"):
        for command in (2, 3):
            env = CommandContactWorld(
                CommandContactConfig(seed=31000, rule=law, rotate=rotation, source=True)
            )
            observations.append(env.reset())
            _, _, _, info = env.step(rotated_command(command, rotation))
            assert info["displacement"] == (0, 0)
            assert info["contact"] and not info["bump"]
            effects[law, command] = info["object_displacement"]
    assert all(np.array_equal(observations[0], obs) for obs in observations)
    assert effects["aligned", 2] == effects["reversed", 3]
    assert effects["aligned", 3] == effects["reversed", 2]
    assert effects["aligned", 2] != effects["aligned", 3]


@pytest.fixture(scope="module")
def acquired():
    return acquire(31001, "aligned")


def test_common_exposure_and_prediction_beyond_body_motion(acquired):
    counts, source = acquired
    assert len(source) == 80
    assert sum(r["contact"] for r in source) == 48
    assert all(set(r["scores"]) == {"conditioned", "action_blind", "empty"} for r in source)
    assert all(r["scores"]["conditioned"]["model_version"] == i for i, r in enumerate(source))
    # Two physical observations per row, with the declared per-row recency decay.
    assert np.allclose(counts.sum(axis=-1)[[0, 1, 4, 5, 8, 9, 12, 13]], 1.95)
    diagnostics = probe(31001, "aligned", counts)
    assert len(diagnostics) == 8
    for key in ("joint_loss", "object_loss"):
        learner = np.mean([r["scores"]["conditioned"][key] for r in diagnostics])
        for baseline in ("action_blind", "shuffled", "empty"):
            assert learner < np.mean([r["scores"][baseline][key] for r in diagnostics])


@pytest.mark.parametrize("rotation", range(4))
def test_swapping_only_command_binding_changes_useful_policy(acquired, rotation):
    counts, _ = acquired
    policies = {}
    for mode in ("conditioned", "shuffled", "action_blind"):
        env = CommandContactWorld(CommandContactConfig(seed=31002, rotate=rotation))
        agent = make_agent(31002, mode, counts)
        agent.observe(env.reset())
        initial = agent.schema.counts.copy()
        for _ in range(3):
            agent.think()
        assert np.array_equal(
            initial, agent.schema.counts
        ), "imagination cannot create empirical evidence"
        policies[mode] = agent.policy[[rotated_command(c, rotation) for c in (2, 3)]]
    assert policies["conditioned"][1] > policies["conditioned"][0]
    assert policies["shuffled"][0] > policies["shuffled"][1]
    np.testing.assert_allclose(policies["action_blind"][0], policies["action_blind"][1], atol=1e-7)


def test_online_controls_learn_final_feedback_while_frozen_retains_counts(acquired):
    counts, _ = acquired
    for mode in ("conditioned", "action_blind", "shuffled", "empty", "frozen", "blind_frozen"):
        agent = make_agent(31003, mode, counts)
        initial = agent.schema.counts.copy()
        row = run_target(CommandContactWorld(CommandContactConfig(seed=31003)), agent)
        assert len(row["transitions"]) == row["steps"] == 2
        if mode in ("frozen", "blind_frozen"):
            assert np.array_equal(initial, agent.schema.counts)
            assert agent.schema.observed == 0
        else:
            assert not np.array_equal(initial, agent.schema.counts)
            assert agent.schema.observed == sum(t["complete"] for t in row["transitions"])
