"""Compute declared contact-family gates and acquisition costs from raw trials."""

import argparse
import json
from pathlib import Path

import numpy as np

from efi.evaluation.interaction import paired


def analyze(data):
    protocol = data["protocol"]
    seeds = list(range(protocol["base_seed"], protocol["base_seed"] + protocol["seeds"]))
    family = [r for r in data["rows"] if r["layout"] in ("west", "north")]
    comparisons = {
        mode: paired(family, "online", mode, "success", seeds)
        for mode in protocol["modes"]
        if mode != "online"
    }
    means = {
        mode: float(np.mean([r["success"] for r in family if r["mode"] == mode]))
        for mode in protocol["modes"]
    }
    training = data["training"]
    contacts = [r for r in training if r["contact"]]
    first_eight, second_eight = [], []
    through_eighth = []
    transitions_through_eighth = []
    for seed in seeds:
        for law in protocol["laws"]:
            source = [r for r in training if r["seed"] == seed and r["law"] == law]
            selected = [r for r in source if r["contact"]]
            first_eight.extend(selected[:8])
            second_eight.extend(selected[8:16])
            transitions_through_eighth.append(selected[7]["exposure"] + 1)
            through_eighth.append(
                sum(r["latency_ms"] for r in source if r["exposure"] <= selected[7]["exposure"])
            )

    def stats(rows):
        return {
            "n": len(rows),
            "log_loss": float(np.mean([r["loss"] for r in rows])),
            "empty_log_loss": float(np.mean([r["empty_loss"] for r in rows])),
            "failed_fraction": float(np.mean([r["bumps"] > 0 for r in rows])),
            "mean_return": float(np.mean([r["return"] for r in rows])),
        }

    keys = ("seed", "law", "layout", "episode")
    observed = ("success", "return", "steps", "contacts", "bumps", "collision", "loss")
    frozen = {
        tuple(r[k] for k in keys): tuple(r[k] for k in observed)
        for r in data["rows"]
        if r["mode"] == "frozen"
    }
    reference = {
        tuple(r[k] for k in keys): tuple(r[k] for k in observed)
        for r in data["rows"]
        if r["mode"] == "tabular"
    }
    return {
        "contact_family": {
            "layouts": ["west", "north"],
            "success": means,
            "online_paired_vs": comparisons,
        },
        "gate_A_behavior_and_prediction": all(
            comparisons[mode]["mean"] >= 0.1 and comparisons[mode]["bootstrap_95"][0] > 0
            for mode in ("empty", "shuffled")
        )
        and data["prediction"]["prequential_log_loss"] < data["prediction"]["empty_log_loss"],
        "source": {
            "all_contacts": stats(contacts),
            "first_eight": stats(first_eight),
            "second_eight": stats(second_eight),
            "mean_cpu_ms_through_eighth_contact": float(np.mean(through_eighth)),
            "mean_source_transitions_through_eighth_contact": float(
                np.mean(transitions_through_eighth)
            ),
            "mean_cpu_ms_per_source_model": sum(r["latency_ms"] for r in training)
            / (len(seeds) * len(protocol["laws"])),
            "total_transitions": len(training),
            "note": "CPU time is measured during the experiment and can include host contention. "
            "Failed means commanded contact left the body in place; "
            "a blocked reaction can be correctly predicted.",
        },
        "frozen_scalar_reference_behavior_identical": frozen == reference,
        "scope": "One bounded contact learner. Gates B/C and integrated preservation remain open. "
        "Scalar reference checks the reducer; it is not an independent architectural baseline.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path, default=Path("runs/interaction/analysis.json"))
    args = parser.parse_args()
    result = analyze(json.loads(args.results.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "contact_family"}, indent=2))


if __name__ == "__main__":
    main()
