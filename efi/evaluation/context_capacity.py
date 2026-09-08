"""Fixed-capacity stress on shared physical streams, with no extra training."""

from dataclasses import asdict
import gzip
import hashlib
import subprocess
import json
from pathlib import Path

import numpy as np

from ..agents.context_memory import ContextMemoryConfig
from .context_memory import common_stream, paired
from .context_replay import replay


def capacity_experiment(seeds=10, base_seed=62000, output=None, progress=False):
    if seeds < 1:
        raise ValueError("positive seed count required")
    root = Path(__file__).resolve().parents[2]
    identity = {
        "base_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "sha256": hashlib.sha256((root / "docs/EFI02_PROTOCOL.md").read_bytes()).hexdigest(),
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ("efi/agents", "efi/core", "efi/envs", "efi/evaluation")
            for p in sorted((root / folder).glob("*.py"))
        },
    }
    streams, trials = [], []
    for seed in range(base_seed, base_seed + seeds):
        for conditions in (2, 3, 5, 8):
            # One field observer records public context, legality, and actual feedback.
            stream = common_stream(seed, modes=("fast",), conditions=conditions, stress=True)
            streams.append(stream)
            for capacity in (1, 3, 6):
                cfg = ContextMemoryConfig(capacity=capacity)
                for mode in ("bank", "no_reuse"):
                    rows, schema = replay(stream, mode, cfg)
                    windows = []
                    for phase in range(conditions, 2 * conditions):
                        contacts = [r for r in rows if r["phase"] == phase and r["contact"]]
                        first = contacts[:8]
                        windows.append(
                            {
                                "phase": phase,
                                "available_contacts": len(contacts),
                                "wrong": (
                                    float(np.mean([r["wrong"] for r in first])) if first else None
                                ),
                                "joint_loss": (
                                    float(np.mean([r["joint_loss"] for r in first]))
                                    if first
                                    else None
                                ),
                            }
                        )
                    trials.append(
                        {
                            "seed": seed,
                            "conditions": conditions,
                            "capacity": capacity,
                            "mode": mode,
                            "windows": windows,
                            "rows": rows,
                            "storage_bytes": schema.storage_bytes,
                            "admissions": schema.admissions,
                            "evictions": schema.evictions,
                        }
                    )
            if progress:
                print(f"[EFI-02 capacity] seed {seed}; {conditions} conditions", flush=True)
    summary = []
    for conditions in (2, 3, 5, 8):
        for capacity in (1, 3, 6):
            selected = [
                r for r in trials if r["conditions"] == conditions and r["capacity"] == capacity
            ]
            values = {
                m: {
                    r["seed"]: np.mean([w["wrong"] for w in r["windows"] if w["wrong"] is not None])
                    for r in selected
                    if r["mode"] == m
                }
                for m in ("bank", "no_reuse")
            }
            bank = [r for r in selected if r["mode"] == "bank"]
            summary.append(
                {
                    "conditions": conditions,
                    "capacity": capacity,
                    "wrong": {m: float(np.mean(list(v.values()))) for m, v in values.items()},
                    "no_reuse_minus_bank": paired(
                        [values["no_reuse"][s] - values["bank"][s] for s in values["bank"]]
                    ),
                    "incomplete_windows": sum(
                        w["available_contacts"] < 8 for r in bank for w in r["windows"]
                    ),
                    "mean_evictions": float(np.mean([r["evictions"] for r in bank])),
                    "storage_bytes": bank[0]["storage_bytes"],
                }
            )
    result = {
        "protocol": {
            **identity,
            "seeds": seeds,
            "base_seed": base_seed,
            "memory": asdict(ContextMemoryConfig()),
            "scope": "One physical stream per seed/load. Replayed bounded schema observers; "
            "no added experience or field execution for each capacity. All contacts charged.",
        },
        "summary": summary,
        "streams": streams,
        "trials": trials,
    }
    if output:
        path = Path(output)
        path.mkdir(parents=True, exist_ok=True)
        with gzip.open(path / "capacity.json.gz", "wt") as handle:
            json.dump(result, handle, default=lambda v: v.tolist(), separators=(",", ":"))
        (path / "capacity_summary.json").write_text(
            json.dumps({k: result[k] for k in ("protocol", "summary")}, indent=2)
        )
    return result
