"""Check EFI-01 gates and paired preservation archives; never rerun experiments."""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value, excluded):
    if isinstance(value, dict):
        return {k: clean(v, excluded) for k, v in value.items() if k not in excluded}
    if isinstance(value, list):
        return [clean(v, excluded) for v in value]
    return value


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate(before, after, data, replay, viewer):
    root = Path(__file__).resolve().parents[1]
    result = json.loads((data / "results.json").read_text())
    profile = json.loads((data / "profile.json").read_text())
    analysis = json.loads((data / "analysis.json").read_text())
    assert all(result["summary"]["research_gates"].values())
    assert all(profile["resource_gates"].values())
    assert analysis["all_pairs_stationary_with_distinct_object_effects"]
    assert analysis["scalar_reference_behavior_identical"]
    assert result["protocol"]["sha256"] == sha(root / "docs/EFI01_PROTOCOL.md")
    assert (len(result["rows"]), len(result["training"]), len(result["diagnostics"])) == (
        8960,
        6400,
        640,
    )
    preserved = {}
    for name, targets, source in (
        ("forage", 212, 0),
        ("crossing", 4800, 0),
        ("transfer", 7680, 400),
        ("contact", 20160, 9600),
    ):
        relative = "forage.json" if name == "forage" else f"{name}/results.json"
        files = [p / relative for p in (before, after)]
        excluded = {"latency_ms", "latency_ms_percentiles"} if name == "contact" else set()
        a, b = [clean(json.loads(p.read_text()), excluded) for p in files]
        assert a == b, f"regression: {name}"
        if name == "forage":
            assert sum(len(v) for v in a.values()) == targets
        else:
            assert len(a["rows"]) == targets
            if source:
                assert len(a["training"]) == source
        preserved[name] = {
            "target_episodes": targets,
            "source_records": source,
            "identical": True,
            "excluded_fields": sorted(excluded),
            "canonical_before_sha256": digest(a),
            "canonical_after_sha256": digest(b),
            "raw_before_sha256": sha(files[0]),
            "raw_after_sha256": sha(files[1]),
        }
    baseline_sources = json.loads((before / "files.json").read_text())
    changed = [p for p, h in baseline_sources.items() if sha(root / p) != h]
    assert changed == ["efi/cli.py"], changed
    tests = {}
    for name, path in (("before", before / "pytest.txt"), ("after", after / "pytest.log")):
        match = re.search(r"(\d+) passed, (\d+) xfailed in ([\d.]+)s", path.read_text())
        assert match, f"missing successful full suite: {path}"
        tests[name] = {
            "passed": int(match[1]),
            "expected_failures": int(match[2]),
            "seconds": float(match[3]),
            "log_sha256": sha(path),
        }
    assert tests["after"]["passed"] >= tests["before"]["passed"] + 10
    rerun = json.loads((replay / "results.json").read_text())
    replay_checks = {}
    for key in ("rows", "training", "diagnostics", "models"):
        first = [r for r in result[key] if r["seed"] == 21000]
        assert clean(first, {"latency_ms", "startup_ms"}) == clean(
            rerun[key], {"latency_ms", "startup_ms"}
        )
        replay_checks[key] = len(first)
    assert json.loads((replay / "episode.json").read_text()) == json.loads(
        (data / "episode.json").read_text()
    )
    assert (
        sha(data / "evaluator_at_run.py.txt")
        == result["protocol"]["source_sha256"]["efi/evaluation/command_contact.py"]
    )
    assert (
        sha(data / "world_at_run.py.txt")
        == result["protocol"]["source_sha256"]["efi/envs/command_contact.py"]
    )
    # Verify that the only world change was its optional default constructor.
    old_world = ast.parse((data / "world_at_run.py.txt").read_text())
    new_world = ast.parse((root / "efi/envs/command_contact.py").read_text())
    for tree in (old_world, new_world):
        cls = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CommandContactWorld"
        )
        cls.body = [
            n for n in cls.body if not isinstance(n, ast.FunctionDef) or n.name != "__init__"
        ]
    assert ast.dump(old_world) == ast.dump(new_world)
    files = [root / p for p in result["protocol"]["source_sha256"]]
    files += [
        root / p
        for p in (
            "efi/evaluation/command_profile.py",
            "tests/test_command_contact.py",
            "scripts/analyze_command_contact.py",
            "scripts/plot_command_contact.py",
            "scripts/validate_command_contact.py",
            "efi/cli.py",
        )
    ]
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_revision": result["protocol"]["base_revision"],
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=root, text=True
        ).strip(),
        "tests": tests,
        "preservation": preserved,
        "previously_existing_executable_or_ignore_files_changed": changed,
        "baseline_source_sha256": baseline_sources,
        "gates": {
            "EFI01_behavior": True,
            "matched_body_prediction": True,
            "resources": True,
            "prior_paths_preserved": True,
            "integrated_successor": "not claimed",
        },
        "presentation_replay": {
            "seed": 21000,
            "identical_populations": replay_checks,
            "all_56_recording_frames_identical": True,
        },
        "viewer": json.loads(viewer.read_text()),
        "post_evaluation_changes": "New evaluator presentation and formatting/JSON helper cleanup: "
        "six chapter groups and selected-command display by default; "
        "default world construction now uses CommandContactConfig. "
        "No agent, world dynamics, sampling, scoring, control or protocol "
        "change. Original evaluator and world sources are archived. First-seed replay "
        "reproduces every behavioral/model record and all 56 recording frames.",
        "final_source_sha256": {str(p.relative_to(root)): sha(p) for p in files},
        "artifact_sha256": {
            str(p.relative_to(root)): sha(p)
            for p in sorted(data.iterdir())
            if p.is_file() and p.name != "validation.json"
        },
        "visual_sha256": {
            p: sha(root / p)
            for p in (
                "docs/assets/interactive/command_contact.html",
                "docs/assets/images/command_contact.png",
                "docs/assets/images/command_contact.pdf",
            )
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("docs/assets/data/command_contact"))
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--viewer", type=Path, required=True)
    args = parser.parse_args()
    data = args.data.resolve()
    validation = validate(args.before, args.after, data, args.replay, args.viewer)
    (data / "validation.json").write_text(json.dumps(validation, indent=2))
    print("EFI-01 gates and all four paired preservation archives passed.")
