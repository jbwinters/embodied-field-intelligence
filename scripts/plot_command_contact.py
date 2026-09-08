"""Render EFI-01's complete archived comparisons without rerunning trials."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


def plot(data, output):
    ink, muted, line, paper = "#263b35", "#64746d", "#d9e0d8", "#f5f5ed"
    colors = {
        "conditioned": "#16816b",
        "action_blind": "#718398",
        "shuffled": "#bd9451",
        "empty": "#985f74",
        "frozen": "#75ad9c",
        "blind_frozen": "#a1adb8",
        "reference": "#acd1c3",
    }
    labels = {
        "conditioned": "Command-conditioned",
        "action_blind": "Command-blind",
        "shuffled": "Swapped commands",
        "empty": "Empty + online",
        "frozen": "Acquired, frozen",
        "blind_frozen": "Blind, frozen",
        "reference": "Scalar reference",
    }
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
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
    fig = plt.figure(figsize=(14, 9.2))
    grid = fig.add_gridspec(
        2,
        2,
        left=0.17,
        right=0.96,
        top=0.79,
        bottom=0.17,
        hspace=0.62,
        wspace=0.38,
        height_ratios=(1.15, 1),
    )
    behavior, prediction, source, target = [
        fig.add_subplot(grid[i, j]) for i in range(2) for j in range(2)
    ]
    fig.text(
        0.055,
        0.946,
        "EMBODIED FIELD INTELLIGENCE   /   EFI-01",
        color=colors["conditioned"],
        fontsize=10,
        weight="bold",
    )
    fig.text(0.055, 0.886, "Same body motion. Different consequences.", fontsize=25, weight="bold")
    fig.text(
        0.055,
        0.84,
        "Experience binds a command to what the object will do. "
        "The existing field learner uses that distinction.",
        fontsize=11,
        color=muted,
    )
    summary = data["summary"]
    modes = list(labels)
    values = [summary["overall"][m]["success"] for m in modes]
    behavior.barh(np.arange(len(modes)), values, color=[colors[m] for m in modes], height=0.62)
    for i, value in enumerate(values):
        behavior.text(
            value + 0.025,
            i,
            f"{value:.1%}",
            va="center",
            fontsize=10,
            weight="bold" if i == 0 else "normal",
        )
    behavior.set_yticks(range(len(modes)), [labels[m] for m in modes])
    behavior.invert_yaxis()
    behavior.set_xlim(0, 1.12)
    behavior.set_xticks([0, 0.5, 1])
    behavior.xaxis.set_major_formatter(PercentFormatter(1))
    behavior.set_title(
        "A   Goal collection · all seven controls", loc="left", weight="bold", pad=14
    )
    probe_modes = modes[:4]
    losses = [summary["prediction"][m]["object_loss"] for m in probe_modes]
    bars = prediction.bar(range(4), losses, color=[colors[m] for m in probe_modes], width=0.58)
    for bar, value in zip(bars, losses):
        prediction.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.04,
            f"{value:.3f}",
            ha="center",
            fontsize=10,
        )
    prediction.set_xticks(range(4), ["Conditioned", "Blind", "Swapped", "Empty"], fontsize=9)
    prediction.set_ylim(0, max(losses) * 1.2)
    prediction.set_ylabel("Object log loss (nats) · lower is better")
    prediction.set_title(
        "B   Predict the block while the body stays still",
        loc="left",
        weight="bold",
        pad=14,
        fontsize=10,
    )
    prediction.text(
        0,
        -0.26,
        "Matched physical interventions; body displacement = (0, 0).",
        transform=prediction.transAxes,
        color=muted,
        fontsize=8.5,
    )
    for m in ("conditioned", "action_blind", "empty"):
        training = data["training"]
        means = [
            np.mean(
                [
                    r["scores"][m]["joint_loss"]
                    for r in training
                    if start <= r["exposure"] < start + 8
                ]
            )
            for start in range(0, 40 * data["protocol"]["acquisition_repetitions"], 8)
        ]
        source.plot(
            np.arange(len(means)) * 8 + 4,
            means,
            color=colors[m],
            linewidth=2,
            marker="o",
            markersize=3,
            label="Empty, frozen" if m == "empty" else labels[m],
        )
    source.set_title("C   Every source attempt is charged", loc="left", weight="bold", pad=14)
    source.set_xlabel("Physical source transitions · eight-attempt bins")
    source.set_ylabel("Joint log loss (nats)")
    source.set_ylim(bottom=0)
    source.set_xticks([0, 40, 80])
    source.legend(frameon=False, fontsize=8, loc="upper right")
    trials = len(summary["learning_curves"]["left_blocked"])
    for m in probe_modes:
        curve = [
            r[m]["success"]
            for layout in ("left_blocked", "right_blocked")
            for r in summary["learning_curves"][layout]
        ]
        target.plot(
            np.arange(1, len(curve) + 1),
            curve,
            color=colors[m],
            linewidth=2,
            marker="o",
            markersize=3,
            label=labels[m],
        )
    target.axvline(trials + 0.5, color=muted, linewidth=0.8, linestyle="--")
    target.set_title("D   Primary controls keep learning", loc="left", weight="bold", pad=14)
    target.set_xlabel("Target trial · left-blocked, then right-blocked")
    target.set_ylabel("Goal collection")
    target.set_ylim(-0.03, 1.08)
    target.yaxis.set_major_formatter(PercentFormatter(1))
    target.set_xticks([1, trials, trials + 1, 2 * trials])
    target.legend(
        frameon=False, fontsize=7.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.29)
    )
    for ax in (behavior, prediction, source, target):
        ax.set_axisbelow(True)
        ax.grid(axis="x" if ax == behavior else "y", color=line, linewidth=0.7)
        ax.tick_params(length=0, pad=6)
    p = data["protocol"]
    fig.text(
        0.055,
        0.065,
        f"{p['seeds']} held-out seeds · {p['target_trials']:,} target trials · "
        f"{p['source_transitions']:,} source transitions · "
        f"{p['diagnostic_interventions']:,} separate probe interventions",
        color=muted,
        fontsize=10,
    )
    fig.text(
        0.055,
        0.033,
        "All failures included. Supplied actuator, geometry, equivariance and two-step horizon. "
        "Frozen and scalar rows are diagnostics. No new agent machinery.",
        color=muted,
        fontsize=8.5,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor=paper)
    fig.savefig(output.with_suffix(".pdf"), facecolor=paper)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path, default=Path("docs/assets/images/command_contact.png"))
    args = parser.parse_args()
    plot(json.loads(args.results.read_text()), args.out)
