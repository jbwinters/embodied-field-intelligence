"""The longer recording is a real continuous stream, not hidden resets."""

import json

import numpy as np

from efi.envs.contact_arena import ContactArena
from efi.evaluation.contact_demo import contact_demo
from efi.visualization.html_viewer import build_payload


def test_collection_replenishes_goal_without_resetting_body_or_object():
    env = ContactArena(steps=9)
    env.reset()
    before_object = env.occupant
    obs, reward, done, info = env.step(0)
    assert reward == 0.99 and info["success"] and not done
    assert env.body == before_object and env.t == 1
    assert env.collections == 1 and env.goals[env.occupant] == 1
    assert env.goals.sum() == 1 and obs.size == 125


def test_response_changes_are_not_observation_labels():
    env = ContactArena(steps=9)
    obs = env.reset()
    env.t = 3
    assert env.rule_at(env.t) == "left"
    np.testing.assert_array_equal(env.observation(), obs)


def test_recording_preserves_every_physical_transition_and_actual_command(tmp_path):
    result = contact_demo(6, 18, output=tmp_path)
    data = json.loads((tmp_path / "episode.json").read_text())
    frames, rows = data["frames"], result["rows"]
    assert len(frames) == 19 and len(rows) == 18
    assert result["source_transitions"] == 0
    assert frames[0]["info"]["learned_transitions"] == 0
    for i, row in enumerate(rows):
        current, following = frames[i]["info"], frames[i + 1]["info"]
        assert current["action"] == row["action"]
        assert current["pos"] == list(row["body_before"])
        assert following["pos"] == list(row["body_after"])
        assert following["step"] == i + 1
        assert following["reward"] == row["reward"]
        if i:
            assert row["body_before"] == rows[i - 1]["body_after"]
            assert row["object_before"] == rows[i - 1]["object_after"]
    assert frames[-1]["info"]["policy"] is frames[-1]["info"]["action"] is None
    assert result["observed_transitions"] == sum(r["complete"] for r in rows)
    payload = build_payload(data)
    assert payload["presentation"]["fps"] == 2
    assert [c["frame"] for c in payload["guide"]["chapters"]] == [0, 6, 12]
