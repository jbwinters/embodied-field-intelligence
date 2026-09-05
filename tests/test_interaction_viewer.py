"""New fields and exact policies extend the established viewer contract."""

import numpy as np

from efi.visualization.html_viewer import build_payload, create_html_viewer


def test_original_fields_keep_the_original_payload_contract():
    data = {
        "frames": [{"GA": np.zeros((5, 5)), "info": {}}],
        "world_frames": [np.zeros((5, 5, 3), dtype=np.uint8)],
    }
    payload = build_payload(data)
    assert set(payload) == {"H", "W", "n", "lmdp", "fields", "world", "walls", "info", "final"}
    assert payload["fields"][0]["label"] == "A scent"


def test_contact_recording_uses_the_original_viewer_and_exact_policy(tmp_path):
    policy = [0.1, 0.2, 0.3, 0.1, 0.3]
    data = {
        "title": "EFI contact learning",
        "frames": [
            {
                "Goal": np.zeros((9, 9)),
                "ActionValue": np.zeros((9, 9)),
                "info": {"lam": 0.02, "policy": policy, "pos": [4, 4], "caption": "Trial one"},
            }
        ],
        "world_frames": [np.zeros((9, 9, 3), dtype=np.uint8)],
    }
    payload = build_payload(data)
    assert payload["info"][0]["policy"] == policy
    assert len(payload["fields"]) == 2
    path = create_html_viewer(data, tmp_path / "episode.html")
    html = open(path).read()
    assert 'id="probeBody"' in html and 'id="modal"' in html
    assert "recorded.policy" in html and "goal_markers" in html
