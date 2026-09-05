"""Plot archived cross-task transfer results without rerunning agents."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path, default=Path("runs/transfer/results.png"))
    args = parser.parse_args()
    result = json.loads(args.results.read_text())
    tasks = ("room", "obstacles", "mixed", "mixed_obstacles")
    modes = ("transfer", "scratch", "exact", "static")
    labels = ("Transferred rules", "Uniform prior", "Exact contexts", "Static forecast")
    colors = ("#16816b", "#647890", "#b58a32", "#ac4d43")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    fig.subplots_adjust(top=0.83, bottom=0.27, wspace=0.28)
    for ax, metric, title in zip(axes, ("success", "return"), ("Interceptions", "Episode return")):
        for i, (mode, label, color) in enumerate(zip(modes, labels, colors)):
            values = [result["summary"][task][mode][metric] for task in tasks]
            ax.bar(np.arange(4) + (i - 1.5) * 0.19, values, 0.18, label=label, color=color)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["Room", "Obstacles", "Hazard", "Obstacles\n+ hazard"])
        ax.set_title(title)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylim(0, 1.05)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].axhline(0, color="#66717a", linewidth=0.7)
    handles, legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("EFI: reusing motion learned during hazard avoidance", fontsize=15)
    p = result["protocol"]
    fig.text(
        0.5,
        0.12,
        f"{p['seeds']} seeds × {p['episodes']} trials per task and contender; "
        "all plotted models frozen during interception",
        ha="center",
        fontsize=10,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
