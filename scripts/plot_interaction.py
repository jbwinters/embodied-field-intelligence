"""Plot the archived contact experiment; never reruns or filters failures."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument(
        "--out", type=Path, default=Path("docs/assets/images/interaction_learning.png")
    )
    args = parser.parse_args()
    data = json.loads(args.results.read_text())
    ink, muted, line, paper = "#263b35", "#64746d", "#d9e0d8", "#f5f5ed"
    teal, blue, ochre = "#16816b", "#718398", "#bd9451"
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "text.color": ink,
            "axes.labelcolor": muted,
            "xtick.color": muted,
            "ytick.color": muted,
            "axes.edgecolor": line,
            "axes.facecolor": paper,
            "figure.facecolor": paper,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig = plt.figure(figsize=(13.6, 7.2))
    grid = fig.add_gridspec(
        2,
        2,
        left=0.065,
        right=0.965,
        top=0.76,
        bottom=0.18,
        width_ratios=(1.14, 1),
        hspace=0.67,
        wspace=0.3,
    )
    success = fig.add_subplot(grid[:, 0])
    learning = fig.add_subplot(grid[0, 1])
    gain = fig.add_subplot(grid[1, 1])
    fig.text(
        0.065,
        0.94,
        "EMBODIED FIELD INTELLIGENCE   /   CONTACT LEARNING",
        color=teal,
        fontsize=10,
        weight="bold",
    )
    fig.text(0.065, 0.877, "Experience changes the approach", fontsize=25, weight="bold")
    fig.text(
        0.065,
        0.825,
        "One locally sensed object. Two-step consequences. No neural network or GPU.",
        fontsize=12,
        color=muted,
    )
    layouts = ("west", "north", "detour")
    for i, (mode, label, color) in enumerate(
        (
            ("online", "Acquired + online", teal),
            ("empty", "Empty evidence", blue),
            ("shuffled", "Shuffled evidence", ochre),
        )
    ):
        values = [data["summary"][layout][mode]["success"] for layout in layouts]
        bars = success.bar(np.arange(3) + (i - 1) * 0.24, values, 0.21, color=color, label=label)
        for bar, value in zip(bars, values):
            success.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.018,
                f"{value:.0%}",
                ha="center",
                fontsize=9,
                color=color,
            )
    success.set_title("A   Goal collection", loc="left", fontweight="bold", pad=16)
    success.set_xticks(range(3), ["West wall", "North wall", "Detour\n(control)"])
    success.set_ylim(0, 1.13)
    success.set_yticks(np.arange(0, 1.01, 0.25))
    success.yaxis.set_major_formatter(PercentFormatter(1))
    success.legend(
        loc="lower left",
        bbox_to_anchor=(-0.02, -0.19),
        ncol=3,
        frameon=False,
        fontsize=8,
        handlelength=1.1,
        columnspacing=1.2,
    )
    training = data["training"]
    exposures = data["protocol"]["source_exposures_per_agent"]
    for key, label, color in (("loss", "Learner", teal), ("empty_loss", "Empty", blue)):
        means = [
            np.mean([r[key] for r in training if start <= r["exposure"] < start + 8])
            for start in range(0, exposures, 8)
        ]
        learning.plot(
            np.arange(len(means)) * 8 + 4,
            means,
            color=color,
            label=label,
            linewidth=2.5,
            marker="o",
            markersize=4,
        )
    learning.set_title("B   Predict first, then learn", loc="left", fontweight="bold", pad=12)
    learning.set_ylabel("Log loss (nats)", fontsize=10)
    learning.set_xlabel("Real source transitions · 8-transition bins", fontsize=9)
    learning.set_ylim(bottom=0)
    learning.set_xticks([0, 40, 80])
    learning.legend(frameon=False, fontsize=9, loc="center right", ncol=2)
    for y, layout in enumerate(layouts):
        p = data["summary"][layout]["paired_vs_empty"]["success"]
        lo, hi = np.asarray(p["bootstrap_95"]) * 100
        mean = p["mean"] * 100
        gain.errorbar(
            mean,
            y,
            xerr=[[mean - lo], [hi - mean]],
            color=teal,
            fmt="o",
            capsize=4,
            markersize=6,
            linewidth=2,
        )
        gain.text(hi + 1.6, y, f"{mean:+.1f} pp", fontsize=10, va="center", color=teal)
    gain.axvline(0, color=muted, linewidth=0.8)
    gain.set_yticks(range(3), ["West wall", "North wall", "Detour"])
    gain.set_ylim(2.6, -0.6)
    gain.set_xlim(-8, 47)
    gain.set_title("C   Gain over empty evidence", loc="left", fontweight="bold", pad=12)
    gain.set_xlabel("Percentage points · paired seed bootstrap 95% intervals", fontsize=9)
    for ax in (success, learning, gain):
        ax.set_axisbelow(True)
        ax.grid(axis="y" if ax != gain else "x", color=line, linewidth=0.7)
        ax.tick_params(length=0, pad=7)
    p = data["protocol"]
    fig.text(
        0.065,
        0.055,
        f"{p['seeds']} held-out seeds  ·  {len(data['rows']):,} target trials  ·  "
        f"{len(training):,} source transitions (80 per acquired model)",
        color=muted,
        fontsize=10,
    )
    fig.text(
        0.065,
        0.024,
        "All failures included. Supplied motor/geometry priors; controlled source interventions. "
        "Action-pooling ablation does not establish an action-conditioning advantage.",
        color=muted,
        fontsize=8.4,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, facecolor=paper)
    plt.close(fig)


if __name__ == "__main__":
    main()
