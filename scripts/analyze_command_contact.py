"""Supplement EFI-01's paired summaries with exposure and failure accounting."""

import argparse
import json
from pathlib import Path

import numpy as np


def analyze(data):
    source, targets, probes = data["training"], data["rows"], data["diagnostics"]
    source_modes = ("conditioned", "action_blind", "empty")
    costs = {
        "physical_transitions": len(source),
        "observer_predictions": sum(len(r["scores"]) for r in source),
        "contacts": sum(r["contact"] for r in source),
        "blocked_attempts": sum(r["bump"] for r in source),
        "lateral_attempts": sum(r["lateral"] for r in source),
        "return": sum(r["return"] for r in source),
        "complete_predictions": sum(s is not None for r in source for s in r["scores"].values()),
    }
    losses = {
        str(rep + 1): {
            m: float(
                np.mean([r["scores"][m]["joint_loss"] for r in source if r["repetition"] == rep])
            )
            for m in source_modes
        }
        for rep in sorted({r["repetition"] for r in source})
    }
    grouped = {}
    for r in probes:
        grouped.setdefault((r["seed"], r["law"], r["layout"], r["front_wall"]), []).append(r)
    matched = all(
        len(pair) == 2
        and {r["command"] for r in pair} == {2, 3}
        and all(r["displacement"] == [0, 0] for r in pair)
        and pair[0]["object_displacement"] != pair[1]["object_displacement"]
        for pair in grouped.values()
    )
    failures = {}
    for mode in data["protocol"]["modes"]:
        rows = [r for r in targets if r["mode"] == mode]
        failures[mode] = {
            "trials": len(rows),
            "failures": sum(not r["success"] for r in rows),
            "collisions": sum(r["collision"] for r in rows),
            "physical_steps": sum(r["steps"] for r in rows),
            "evidence_updates": (
                sum(t["complete"] for r in rows for t in r["transitions"])
                if mode not in ("frozen", "blind_frozen")
                else 0
            ),
            "unsuccessful_first_command": sum(
                r["transitions"][0]["object_displacement"] == [0, 0]
                for r in rows
                if not r["success"]
            ),
        }
    online = [r for r in targets if r["mode"] == "conditioned"]
    reference = [r for r in targets if r["mode"] == "reference"]
    fields = ("success", "return", "steps", "contacts", "bumps", "collision")
    reference_equal = all(
        all(a[k] == b[k] for k in fields)
        and all(x["action"] == y["action"] for x, y in zip(a["transitions"], b["transitions"]))
        for a, b in zip(online, reference)
    )
    policy_error = max(
        abs(x - y)
        for a, b in zip(online, reference)
        for t, u in zip(a["transitions"], b["transitions"])
        for x, y in zip(t["policy"], u["policy"])
    )
    return {
        "source_cost": costs,
        "source_prequential_joint_loss_by_repetition": losses,
        "matched_probe_pairs": len(grouped),
        "all_pairs_stationary_with_distinct_object_effects": matched,
        "prediction_interventions": len(probes),
        "diagnostic_predictions": sum(len(r["scores"]) for r in probes),
        "target_failures_and_learning": failures,
        "scalar_reference_behavior_identical": reference_equal,
        "scalar_reference_max_policy_difference": policy_error,
        "max_outcome_terms": max(t["outcome_terms"] for r in targets for t in r["transitions"]),
        "note": "Source failures are failed commanded effects, not failed reward tasks: "
        "source has no goals. "
        "Probe observations are isolated evaluation interventions, not target training. "
        "Complete target feedback is charged once per actual transition; "
        "frozen diagnostics score but do not learn.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path, default=Path("runs/command-contact/analysis.json"))
    args = parser.parse_args()
    result = analyze(json.loads(args.results.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
