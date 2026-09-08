"""Audit archived EFI-02 evidence, preservation, and current source identity."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from efi.evaluation.context_memory import summarize
from efi.evaluation.context_replay import replay


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items() if k not in {"latency_ms", "startup_ms"}}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate(data, baseline, rerun, tests, viewer):
    root = Path(__file__).resolve().parents[1]
    with gzip.open(data / "results.json.gz", "rt") as handle:
        result = json.load(handle)
    with gzip.open(data / "capacity.json.gz", "rt") as handle:
        capacity = json.load(handle)
    summary = summarize(result["common"], result["behavior"])
    assert json.loads(json.dumps(summary)) == result["summary"]
    assert all(isinstance(v, bool) for v in summary["research_gates"].values())
    assert result["protocol"]["sha256"] == sha(root / "docs/EFI02_PROTOCOL.md")
    assert len(result["common"]) == 80 and len(result["behavior"]) == 400
    for p, h in result["protocol"]["source_sha256"].items():
        assert sha(root / p) == h, f"source changed after evaluation: {p}"
    for delay in (0, 3):
        assert sorted(s["seed"] for s in result["common"] if s["delay"] == delay) == list(
            range(61000, 61040)
        )
    for stream in result["common"]:
        assert len(stream["rows"]) == sum(stream["lengths"])
        for r in stream["rows"]:
            assert r["memory"]["bank"] == r["memory"]["no_reuse"]
            assert r["memory"]["bank"]["active"] <= 3
    replay_checks = {}
    for stream in result["common"][:2]:
        original = {r["tick"]: r for r in stream["rows"]}
        for mode in result["protocol"]["modes"]:
            rows, schema = replay(stream, mode)
            for r in rows:
                expected = original[r["tick"]]["scores"][mode]
                assert r["wrong"] == expected["wrong"]
                assert r["joint_loss"] == expected["joint_loss"]
            assert schema.observed == stream["final_memory"][mode]["observed"]
            replay_checks[f"{stream['seed']}/{stream['delay']}/{mode}"] = len(rows)
    assert len(capacity["streams"]) == 40 and len(capacity["trials"]) == 240
    assert all(r["incomplete_windows"] == 0 for r in capacity["summary"])
    profile = json.loads((data / "profile.json").read_text())
    assert all(profile["resource_gates"].values())
    intervention = json.loads((data / "intervention.json").read_text())
    for r in intervention["rows"]:
        assert r["intact"][1] > r["intact"][0] and r["swapped"][0] > r["swapped"][1]
        assert np.isclose(r["erased"][0], r["erased"][1])
    before = json.loads((root / "docs/assets/data/command_contact/results.json").read_text())
    after = json.loads((rerun / "results.json").read_text())
    preserved = {}
    for key in ("rows", "training", "diagnostics", "models"):
        a, b = clean(before[key]), clean(after[key])
        assert a == b, key
        preserved[key] = {
            "records": len(a),
            "identical": True,
            "before_sha256": digest(a),
            "after_sha256": digest(b),
        }
    sources = json.loads((baseline / "files.json").read_text())
    changed = [p for p, h in sources.items() if sha(root / p) != h]
    assert changed == ["efi/cli.py"], changed
    test_records = {}
    for name, path in (("before", baseline / "pytest.log"), ("after", tests)):
        match = re.search(r"(\d+) passed, (\d+) xfailed in ([\d.]+)s", path.read_text())
        assert match, str(path)
        test_records[name] = {
            "passed": int(match[1]),
            "xfailed": int(match[2]),
            "seconds": float(match[3]),
            "sha256": sha(path),
        }
    assert test_records["after"]["passed"] >= test_records["before"]["passed"] + 9
    with gzip.open(data / "episode.json.gz", "rt") as handle:
        episode = json.load(handle)
    first = next(
        b
        for b in result["behavior"]
        if b["seed"] == 61000 and b["mode"] == "bank" and b["delay"] == 0
    )
    assert len(episode["frames"]) == first["steps"] + 1
    validation = {
        "experiment_integrity_passed": True,
        "capability_demonstrated": all(summary["research_gates"].values()),
        "failed_capability_gates": [k for k, v in summary["research_gates"].items() if not v],
        "gates": summary["research_gates"],
        "resources": profile["resource_gates"],
        "tests": test_records,
        "preservation": preserved,
        "unchanged_old_paths": "All preexisting executable/test files "
        "except additive CLI registration.",
        "integrated_successor": "Not claimed; existing controllers/defaults preserved.",
        "excluded_preservation_fields": ["latency_ms", "startup_ms"],
        "baseline_source_sha256": sources,
        "field_replay_identical": replay_checks,
        "common_physical_steps": sum(len(s["rows"]) for s in result["common"]),
        "behavior_physical_steps": sum(s["steps"] for s in result["behavior"]),
        "capacity_physical_steps": sum(len(s["rows"]) for s in capacity["streams"]),
        "intervention_physical_steps": intervention["source_steps"],
        "full_first_lifetime_frames": len(episode["frames"]),
        "viewer": json.loads(viewer.read_text()),
        "artifact_sha256": {
            str(p.relative_to(data)): sha(p)
            for p in sorted(data.rglob("*"))
            if p.is_file() and p.name != "validation.json"
        },
    }
    (data / "validation.json").write_text(json.dumps(validation, indent=2))
    return validation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("docs/assets/data/context_memory"))
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--rerun", type=Path, required=True)
    parser.add_argument("--tests", type=Path, required=True)
    parser.add_argument("--viewer", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.data.resolve(), args.baseline, args.rerun, args.tests, args.viewer)
    print("Evidence integrity, capacity coverage, resources, and preservation audited.")
    print(
        f"Capability demonstrated: {result['capability_demonstrated']}; "
        f"failed gates: {result['failed_capability_gates']}"
    )
