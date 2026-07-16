#!/usr/bin/env python3
"""Kappa (thinking-rate) experiment: behavior vs value-sweeps-per-tick.

Produces docs/assets/data/kappa_curves.json and a plot. With --deep-verify,
also measures fixed-point tracking error against the contraction bound
eps * gamma^K / (1 - gamma^K) on one instrumented episode.

Usage:
    python scripts/exp_kappa.py [--episodes 30] [--seeds 3] [--deep-verify]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from efi.configs import AgentConfig, Ablations, EnvConfig
from efi.envs import ForageWorld
from efi.agents import FieldController, ForageAdapter
from efi.core.desirability import value_sweeps
from efi.evaluation import run_experiment, run_episode

KAPPAS = [0, 1, 2, 3, 5, 8]
SIZES = [15, 30]


def success_rate(results, nA):
    ms = results.metrics
    return sum(1 for m in ms if m.targets_collected.get("A", 0) >= nA) / len(ms)


def run_curves(episodes, seeds):
    out = []
    for H in SIZES:
        for kappa in KAPPAS:
            env_cfg = EnvConfig(H=H, W=H, max_steps=int(0.9 * H * H))
            agent_cfg = AgentConfig(valA_init=1.0, z_sweeps=kappa)
            res = run_experiment(env_cfg, agent_cfg, None, Ablations(schema=0),
                                 episodes=episodes, seeds=seeds, use_controller=True)
            row = {
                "size": H,
                "kappa": kappa,
                "mean_return": res.mean_return,
                "std_return": res.std_return,
                "success": success_rate(res, env_cfg.n_targets_A),
                "mean_steps": res.mean_steps,
                "mean_residual": float(np.mean([m.mean_residual for m in res.metrics])),
                "gamma_hat_median": float(np.median(
                    [m.gamma_hat_median for m in res.metrics if m.gamma_hat_median > 0] or [0.0])),
            }
            out.append(row)
            print(f"[kappa] {H}x{H} kappa={kappa}: success={row['success']:.1%} "
                  f"return={row['mean_return']:+.3f} resid={row['mean_residual']:.4f}")
    return out


def deep_verify(H=15, every=10):
    """Instrumented episode: measure |V_t - V_inf_t| and compare to the
    steady-state contraction bound eps * gamma^K / (1 - gamma^K)."""
    env = ForageWorld(EnvConfig(H=H, W=H, max_steps=int(0.9 * H * H), seed=7))
    env.reset()
    cfg = AgentConfig(valA_init=1.0, z_sweeps=3, seed=7)
    agent = FieldController(env, ForageAdapter(env), cfg, Ablations(schema=0), seed=7)

    errors, eps_drift, gammas = [], [], []
    V_inf_prev = None

    original = agent.compose_value

    def instrumented(**kw):
        nonlocal V_inf_prev
        V = original(**kw)
        instrumented.calls += 1
        if instrumented.calls % every == 0:
            V_inf, _ = value_sweeps(V.copy(), agent.last_q, agent.last_R_inj,
                                    agent.last_walls_used, lam=agent.lam_current,
                                    sweeps=200)
            passable = ~agent.last_walls_used
            errors.append(float(np.abs(V_inf[passable] - V[passable]).max()))
            if V_inf_prev is not None and V_inf_prev.shape == V_inf.shape:
                eps_drift.append(float(np.abs(V_inf[passable] - V_inf_prev[passable]).max()))
            V_inf_prev = V_inf
        res = agent.last_residuals
        if len(res) >= 2 and res[-2] > 1e-12:
            gammas.append(res[-1] / res[-2])
        return V

    instrumented.calls = 0
    agent.compose_value = instrumented
    run_episode(env, agent, None, Ablations(schema=0))

    gamma = float(np.median(gammas)) if gammas else 0.0
    eps = float(np.median(eps_drift)) if eps_drift else 0.0
    K = cfg.z_sweeps
    bound = eps * (gamma ** K) / max(1e-9, 1.0 - gamma ** K) if gamma < 1 else float("inf")

    # The contraction bound is a STEADY-STATE bound: it assumes the fixed
    # point drifts by ~eps per window. Pickups and wall/target discoveries
    # JUMP the fixed point discontinuously; those windows are transients the
    # bound does not (and should not) cover. Split windows by measured
    # drift: quiescent = drift <= 3x median drift.
    paired = list(zip(errors[1:], eps_drift))  # error_i vs drift into window i
    thresh = 3.0 * eps if eps > 0 else float("inf")
    quiescent = [(e, d) for (e, d) in paired if d <= thresh]
    transient = [(e, d) for (e, d) in paired if d > thresh]
    within_q = [e <= max(bound, 1e-6) for (e, _) in quiescent]

    report = {
        "gamma_hat_median": gamma,
        "eps_fixed_point_drift_per_%d_ticks" % every: eps,
        "eps_per_tick": eps / every,
        "steady_state_bound_window": bound,
        "tracking_errors": errors,
        "median_tracking_error": float(np.median(errors)) if errors else 0.0,
        "n_windows": len(paired),
        "n_quiescent": len(quiescent),
        "n_transient": len(transient),
        "fraction_within_bound_quiescent": float(np.mean(within_q)) if within_q else 1.0,
        "median_error_transient": float(np.median([e for (e, _) in transient])) if transient else 0.0,
    }
    print(f"[deep-verify] gamma={gamma:.3f} eps/window={eps:.4f} bound={bound:.4f} "
          f"median_err={report['median_tracking_error']:.4f} | "
          f"quiescent windows within bound: {report['fraction_within_bound_quiescent']:.0%} "
          f"({report['n_quiescent']}/{report['n_windows']}); "
          f"{report['n_transient']} transient (belief-jump) windows excluded")
    return report


def plot(rows, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[kappa] matplotlib unavailable; skipping plot")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for H, ax_i, metric in [(15, 0, "success"), (30, 1, "success")]:
        ax = axes[ax_i]
        sub = [r for r in rows if r["size"] == H]
        ax.plot([r["kappa"] for r in sub], [r[metric] for r in sub], "o-")
        ax.set_title(f"{H}x{H}")
        ax.set_xlabel("kappa (value sweeps per tick)")
        ax.set_ylabel("success rate")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Thinking rate vs performance")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"[kappa] plot saved to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--deep-verify", action="store_true")
    ap.add_argument("--deep-only", action="store_true",
                    help="run only the deep verification, merge into existing json")
    args = ap.parse_args()

    out_dir = Path("docs/assets/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "kappa_curves.json"

    if args.deep_only:
        payload = json.loads(json_path.read_text()) if json_path.exists() else {}
        payload["deep_verify"] = deep_verify()
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[kappa] deep-verify merged into {json_path}")
        return

    rows = run_curves(args.episodes, args.seeds)
    payload = {"curves": rows,
               "config": {"episodes": args.episodes, "seeds": args.seeds,
                          "kappas": KAPPAS, "sizes": SIZES}}
    if args.deep_verify:
        payload["deep_verify"] = deep_verify()

    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[kappa] data saved to {json_path}")

    Path("docs/assets/images").mkdir(parents=True, exist_ok=True)
    plot(rows, "docs/assets/images/kappa_curves.png")


if __name__ == "__main__":
    main()
