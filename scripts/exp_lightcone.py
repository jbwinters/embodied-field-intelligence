#!/usr/bin/env python3
"""Speed-of-thought experiment: measure the internal light cone.

Reveal a target at distance d for one tick; count think-ticks until the
policy responds. Expect latency ~ a * d / kappa + b. A non-local shortcut
would make latency flat in d.

Everything is deterministic (exact softmax distributions, no env stepping),
so no repetitions are needed.

Usage: python scripts/exp_lightcone.py
Outputs: docs/assets/data/lightcone.json, docs/assets/images/lightcone.png,
and a section appended to docs/TRACKING.md by hand (see stdout).
"""

import json
from pathlib import Path

import numpy as np

from efi.evaluation.probes import reaction_latency

DS = [5, 10, 15, 20]
KAPPAS = [1, 3, 5]


def main():
    rows = []
    for kappa in KAPPAS:
        for d in DS:
            lat = reaction_latency(d, kappa)
            rows.append({"d": d, "kappa": kappa,
                         "latency": (-1 if lat is None else lat)})
            print(f"[lightcone] d={d:2d} kappa={kappa}: latency="
                  f"{'never' if lat is None else lat}")

    # Fit latency ~ a * (d/kappa) + b over measured points
    pts = [(r["d"] / r["kappa"], r["latency"]) for r in rows if r["latency"] >= 0]
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    a, b = np.polyfit(xs, ys, 1)
    pred = a * xs + b
    ss_res = float(np.sum((ys - pred) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    print(f"[lightcone] latency ~ {a:.2f} * d/kappa + {b:.2f}   R^2={r2:.3f}")
    print(f"[lightcone] internal speed of thought c = {1.0 / a:.2f} cells "
          f"per sweep" if a > 0 else "")

    out_dir = Path("docs/assets/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "lightcone.json", "w") as f:
        json.dump({"rows": rows, "fit": {"a": float(a), "b": float(b),
                                         "r2": float(r2)}}, f, indent=2)
    print(f"[lightcone] data saved to {out_dir / 'lightcone.json'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for kappa in KAPPAS:
            sub = [(r["d"], r["latency"]) for r in rows
                   if r["kappa"] == kappa and r["latency"] >= 0]
            ax.plot([p[0] for p in sub], [p[1] for p in sub], "o-",
                    label=f"kappa={kappa}")
        ax.set_xlabel("stimulus distance d (cells)")
        ax.set_ylabel("reaction latency (think-ticks)")
        ax.set_title(f"The internal light cone (fit: {a:.2f}·d/κ + {b:.2f}, "
                     f"R²={r2:.3f})")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        Path("docs/assets/images").mkdir(parents=True, exist_ok=True)
        fig.savefig("docs/assets/images/lightcone.png", dpi=120)
        print("[lightcone] plot saved to docs/assets/images/lightcone.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
