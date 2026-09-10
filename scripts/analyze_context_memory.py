"""Describe EFI-02 behavioral losses without changing the locked experiment."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np


def longest(flags):
    best = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def analyze(data):
    result = {
        "scope": "Descriptive trajectory analysis after held-out evaluation; "
        "not a causal isolation or parameter-selection experiment.",
        "delays": {},
    }
    for delay in (0, 3):
        runs = [r for r in data["behavior"] if r["delay"] == delay]
        modes = {
            m: {r["seed"]: r for r in runs if r["mode"] == m} for m in data["protocol"]["modes"]
        }
        summary = {}
        for mode, records in modes.items():
            summary[mode] = {
                "phases_with_fewer_than_eight_contacts": sum(
                    sum(t["contact"] for t in r["rows"] if t["phase"] == phase) < 8
                    for r in records.values()
                    for phase in range(5)
                ),
                "return_phases_with_fewer_than_eight_contacts": sum(
                    sum(t["contact"] for t in r["rows"] if t["phase"] == phase) < 8
                    for r in records.values()
                    for phase in (2, 4)
                ),
                "mean_longest_no_collection_run": float(
                    np.mean(
                        [longest(not t["success"] for t in r["rows"]) for r in records.values()]
                    )
                ),
                "mean_longest_no_contact_run": float(
                    np.mean(
                        [longest(not t["contact"] for t in r["rows"]) for r in records.values()]
                    )
                ),
                "mean_longest_wait_run": float(
                    np.mean(
                        [longest(t["action"] == 4 for t in r["rows"]) for r in records.values()]
                    )
                ),
                "mean_final_active_archives": float(
                    np.mean(
                        [
                            (
                                r["rows"][-1]["memory"]["active"]
                                if mode not in ("fast", "slow")
                                else 0
                            )
                            for r in records.values()
                        ]
                    )
                ),
                "mean_admissions": float(
                    np.mean([r["rows"][-1]["memory"]["admissions"] for r in records.values()])
                ),
                "mean_evictions": float(
                    np.mean([r["rows"][-1]["memory"]["evictions"] for r in records.values()])
                ),
                "mean_collections_by_phase": [
                    float(
                        np.mean(
                            [
                                sum(t["success"] for t in r["rows"] if t["phase"] == p)
                                for r in records.values()
                            ]
                        )
                    )
                    for p in range(5)
                ],
            }
        differences = {
            seed: r["collections"] - modes["fast"][seed]["collections"]
            for seed, r in modes["bank"].items()
        }
        result["delays"][str(delay)] = {
            "modes": summary,
            "bank_vs_fast_seed_outcomes": {
                "more": sum(v > 0 for v in differences.values()),
                "same": sum(v == 0 for v in differences.values()),
                "fewer": sum(v < 0 for v in differences.values()),
            },
            "collections_difference_by_seed": differences,
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    args = parser.parse_args()
    with gzip.open(args.data / "results.json.gz", "rt") as handle:
        result = analyze(json.load(handle))
    (args.data / "analysis.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
