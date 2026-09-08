"""Plot the complete EFI-02 archive in the project's existing visual style."""

import argparse
import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

COLORS = {
    "bank": "#16816b",
    "fast": "#718398",
    "slow": "#bd9451",
    "no_reuse": "#985f74",
    "no_history": "#75ad9c",
}
LABELS = {
    "bank": "Context bank",
    "fast": "Existing fast",
    "slow": "Slow table",
    "no_reuse": "Archives disconnected",
    "no_history": "No history",
}
PAPER, INK, MUTED, LINE = "#f5f5ed", "#263b35", "#64746d", "#d9e0d8"


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.edgecolor": LINE,
            "axes.facecolor": PAPER,
            "figure.facecolor": PAPER,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def finish(fig, axes, output):
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=LINE, linewidth=0.7)
        ax.tick_params(length=0, pad=6)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor=PAPER)
    fig.savefig(output.with_suffix(".pdf"), facecolor=PAPER)
    plt.close(fig)


def seed_interval(values):
    values = np.asarray(values)
    means = np.random.RandomState(29).choice(values, (10000, len(values))).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def plot(data, output):
    style()
    fig = plt.figure(figsize=(14, 10))
    grid = fig.add_gridspec(
        2, 2, left=0.09, right=0.96, top=0.79, bottom=0.17, hspace=0.62, wspace=0.28
    )
    axes = [fig.add_subplot(grid[i, j]) for i in range(2) for j in range(2)]
    fig.text(
        0.055,
        0.945,
        "EMBODIED FIELD INTELLIGENCE   /   EFI-02",
        color=COLORS["bank"],
        weight="bold",
    )
    fig.text(0.055, 0.888, "Better recall, an unfinished milestone.", fontsize=25, weight="bold")
    fig.text(
        0.055,
        0.84,
        "Returning responses are easier to predict. The autonomous behavior guardrail is not met.",
        fontsize=11,
        color=MUTED,
    )
    summary = data["summary"]
    for ax, name, title in zip(
        axes[:2],
        ("return", "new"),
        ("A   A familiar condition returns", "B   A new condition appears"),
    ):
        values = [summary["common"]["0"][name]["modes"][m]["wrong"] for m in LABELS]
        intervals = np.asarray(
            [
                seed_interval(
                    [v[m]["wrong"] for v in summary["common"]["0"][name]["per_seed"].values()]
                )
                for m in LABELS
            ]
        )
        bars = ax.bar(
            range(5),
            values,
            color=list(COLORS.values()),
            width=0.63,
            yerr=np.array([values - intervals[:, 0], intervals[:, 1] - values]),
            error_kw={"ecolor": INK, "elinewidth": 0.9, "capsize": 3},
        )
        for b, v, ci in zip(bars, values, intervals):
            ax.text(
                b.get_x() + b.get_width() / 2, ci[1] + 0.025, f"{v:.1%}", ha="center", fontsize=10
            )
        ax.set_xticks(range(5), ["Bank", "Fast", "Slow", "No reuse", "No history"], fontsize=9)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylabel("Wrong predictions · first eight contacts")
        ax.set_title(title, loc="left", weight="bold", pad=16)
    ax = axes[2]
    rows = [
        r
        for s in data["common"]
        if s["delay"] == 0
        for r in s["rows"]
        if r["returning"] and r["contact"] and r["contact_index"] < 8
    ]
    for m in LABELS:
        curve = [
            np.mean([r["scores"][m]["wrong"] for r in rows if r["contact_index"] == i])
            for i in range(8)
        ]
        ax.plot(
            np.arange(1, 9), curve, color=COLORS[m], lw=2, marker="o", markersize=3, label=LABELS[m]
        )
    ax.set_title("C   Recognition follows real feedback", loc="left", weight="bold", pad=16)
    ax.set_xlabel("Contact after return · prediction before learning")
    ax.set_ylabel("Wrong joint-effect prediction")
    ax.set_ylim(-0.03, 1.03)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xticks(range(1, 9))
    ax = axes[3]
    for i, m in enumerate(LABELS):
        vals = [summary["behavior"][str(d)]["modes"][m]["collection_rate"] for d in (0, 3)]
        intervals = np.asarray(
            [
                seed_interval(
                    [
                        r["collections"] / r["steps"]
                        for r in data["behavior"]
                        if r["mode"] == m and r["delay"] == d
                    ]
                )
                for d in (0, 3)
            ]
        )
        ax.bar(
            np.array([0, 1]) + (i - 2) * 0.145,
            vals,
            width=0.13,
            color=COLORS[m],
            label=LABELS[m],
            yerr=np.array([vals - intervals[:, 0], intervals[:, 1] - vals]),
            error_kw={"ecolor": INK, "elinewidth": 0.8, "capsize": 2},
        )
    ax.set_title("D   Separate autonomous lifetimes", loc="left", weight="bold", pad=16)
    ax.set_xticks([0, 1], ["Immediate feedback", "Three neutral contacts"])
    ax.set_ylabel("Goals collected per physical tick")
    ax.set_ylim(0, 0.6)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.083),
        ncol=5,
        frameon=False,
        fontsize=9,
    )
    fig.text(
        0.055,
        0.052,
        f"{data['protocol']['seeds']} held-out seeds · all five controls · "
        "uninterrupted empty-start lifetimes · every physical step charged",
        color=MUTED,
        fontsize=10,
    )
    fig.text(
        0.055,
        0.025,
        "Whiskers: seed-bootstrap 95% intervals. Supplied effects and actuator physics. "
        "Recall is not condition-sequence prediction.",
        color=MUTED,
        fontsize=9,
    )
    finish(fig, axes, output)


def capacity_plot(data, output):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.7))
    fig.subplots_adjust(left=0.09, right=0.96, bottom=0.22, top=0.72, wspace=0.32)
    fig.text(0.055, 0.91, "EFI-02   /   FINITE MEMORY", weight="bold", color=COLORS["bank"])
    fig.text(0.055, 0.81, "Retention has a capacity limit.", weight="bold", fontsize=23)
    for capacity, color in zip((1, 3, 6), (COLORS["fast"], COLORS["bank"], COLORS["no_reuse"])):
        rows = [r for r in data["summary"] if r["capacity"] == capacity]
        x = [r["conditions"] for r in rows]
        axes[0].plot(
            x,
            [100 * r["no_reuse_minus_bank"]["mean"] for r in rows],
            color=color,
            marker="o",
            lw=2,
            label=f"{capacity} archive slots",
        )
        axes[0].fill_between(
            x,
            [100 * r["no_reuse_minus_bank"]["bootstrap_95"][0] for r in rows],
            [100 * r["no_reuse_minus_bank"]["bootstrap_95"][1] for r in rows],
            color=color,
            alpha=0.12,
        )
        axes[1].plot(x, [r["mean_evictions"] for r in rows], color=color, marker="o", lw=2)
    axes[0].axhline(0, color=MUTED, ls="--", label="Archives disconnected")
    axes[0].set_ylabel("Error reduction vs no reuse · percentage points")
    axes[0].set_ylim(-12, 50)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_ylabel("Archive evictions per lifetime")
    axes[1].set_ylim(bottom=0)
    for ax in axes:
        ax.set_xlabel("Distinct supplied response conditions")
        ax.set_xticks((2, 3, 5, 8))
    fig.text(
        0.055,
        0.08,
        "One shared physical stream per seed/load. Replayed observers get no extra experience; "
        "every return window is retained.",
        fontsize=9,
        color=MUTED,
    )
    finish(fig, axes, output)


def timeline_plot(data, output):
    style()
    stream = next(s for s in data["common"] if s["delay"] == 0)
    seed = stream["seed"]
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.subplots_adjust(left=0.09, right=0.96, bottom=0.13, top=0.75, hspace=0.42)
    fig.text(
        0.055,
        0.944,
        "EFI-02   /   EVERY CHANGE, EVERY PHYSICAL STEP",
        color=COLORS["bank"],
        weight="bold",
    )
    fig.text(0.055, 0.88, "Learning through every change.", fontsize=25, weight="bold")
    fig.text(
        0.055,
        0.838,
        f"First held-out seed {seed}, selected before its outcome. "
        "Response labels below are evaluator truth.",
        color=MUTED,
        fontsize=10,
    )
    contacts = [r for r in stream["rows"] if r["contact"]]
    for mode in ("bank", "fast", "no_reuse"):
        errors = [r["scores"][mode]["wrong"] for r in contacts]
        curve = [np.mean(errors[max(0, i - 7) : i + 1]) for i in range(len(errors))]
        axes[0].plot(
            [r["tick"] for r in contacts], curve, color=COLORS[mode], lw=1.6, label=LABELS[mode]
        )
    axes[0].set_title(
        "A   Common experience · error over the latest eight contacts",
        loc="left",
        weight="bold",
        pad=28,
    )
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylabel("Prediction error")
    axes[0].legend(frameon=False, fontsize=8, ncol=3, loc="upper right")
    weights = np.asarray([r["memory"]["bank"]["weights"] for r in stream["rows"]])
    axes[1].stackplot(
        np.arange(len(weights)),
        weights.T,
        labels=("Fast", "Archive slot 1", "Archive slot 2", "Archive slot 3"),
        colors=(COLORS["fast"], COLORS["bank"], COLORS["slow"], COLORS["no_reuse"]),
    )
    axes[1].set_title(
        "B   Common experience · applicability after actual feedback",
        loc="left",
        weight="bold",
        pad=14,
    )
    axes[1].set_ylim(0, 1)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_ylabel("Posterior weight")
    axes[1].legend(
        frameon=True,
        facecolor=PAPER,
        edgecolor=LINE,
        framealpha=0.9,
        fontsize=8,
        ncol=4,
        loc="upper right",
    )
    for mode in ("bank", "fast", "no_history"):
        run = next(
            r
            for r in data["behavior"]
            if r["seed"] == seed and r["delay"] == 0 and r["mode"] == mode
        )
        axes[2].plot(
            np.arange(run["steps"]),
            np.cumsum([r["success"] for r in run["rows"]]),
            color=COLORS[mode],
            lw=2,
            label=LABELS[mode],
        )
    axes[2].set_title(
        "C   Separate autonomous lifetimes · each learns from its own actions",
        loc="left",
        weight="bold",
        pad=14,
    )
    axes[2].set_ylabel("Cumulative goals")
    axes[2].set_xlabel("Physical tick · no resets at response changes")
    axes[2].legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    ends = np.cumsum(stream["lengths"])
    for ax in axes:
        for boundary in ends[:-1]:
            ax.axvline(boundary, color=MUTED, ls="--", lw=0.8, alpha=0.7)
        ax.set_xlim(0, ends[-1])
    for start, end, condition in zip(np.r_[0, ends[:-1]], ends, stream["order"]):
        axes[0].text(
            (start + end) / 2,
            1.055,
            f"Response {condition+1}",
            ha="center",
            transform=axes[0].get_xaxis_transform(),
            fontsize=8,
            color=MUTED,
        )
    fig.text(
        0.055,
        0.052,
        "Archive slots can be replaced; colors are storage slots, "
        "not supplied condition identities. "
        "The source curriculum and autonomous runs are separate.",
        fontsize=9,
        color=MUTED,
    )
    finish(fig, axes, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--out", type=Path, default=Path("docs/assets/images/context_memory.png"))
    args = parser.parse_args()
    with gzip.open(args.data / "results.json.gz", "rt") as handle:
        data = json.load(handle)
        plot(data, args.out)
        timeline_plot(data, args.out.with_name("context_timeline.png"))
    with gzip.open(args.data / "capacity.json.gz", "rt") as handle:
        capacity_plot(json.load(handle), args.out.with_name("context_capacity.png"))
