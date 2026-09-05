"""Plot the paired crossing benchmark without rerunning the simulation.

python scripts/plot_crossing.py runs/predictive-crossing/results.json \
    --out docs/assets/images/predictive_crossing.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path, default=Path("runs/crossing/results.png"))
    args = parser.parse_args()
    payload = json.loads(args.results.read_text())
    phases = ("acquire", "transfer", "reverse")
    modes = ("learned", "static", "unlearned", "frozen")
    labels = ("Learns online", "Static forecast", "Unlearned dynamics", "Frozen at reversal")
    colors = ("#16816b", "#af4a40", "#6b7891", "#b37a26")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    fig.subplots_adjust(top=0.83, bottom=0.28, wspace=0.25)
    for ax, metric, title in zip(
        axes, ("success", "collision"), ("Successful crossings", "Collisions")
    ):
        for i, (mode, label, color) in enumerate(zip(modes, labels, colors)):
            values = [payload["summary"][p][mode][metric] for p in phases]
            ax.bar(np.arange(3) + (i - 1.5) * 0.19, values, 0.18, color=color, label=label)
        ax.set_xticks(range(3), ["Learn\n9×9", "Transfer\n11×13", "Reverse\nmotion rule"])
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_title(title)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    protocol = payload["protocol"]
    fig.suptitle("EFI: learned forecasts improve safe progress", fontsize=15)
    axes[0].set_ylabel("Fraction of trials")
    fig.text(
        0.5,
        0.12,
        f"{protocol['seeds']} independent seeds × "
        f"{protocol['episodes_per_phase']} trials per phase; "
        f"same {protocol['horizon']}-step planning horizon and 5×5 sensing",
        fontsize=10,
        ha="center",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
