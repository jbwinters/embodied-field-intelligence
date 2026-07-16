#!/usr/bin/env python3
"""Run all baselines + EFI on the identical environment distribution and
emit the normalized-score table.

Writes docs/assets/data/baselines.json and prints a markdown table.
Normalized score = (X - random) / (astar - random).

Usage:
    python scripts/make_baseline_table.py [--episodes 40] [--seeds 5]
        [--q-train-episodes 2000] [--H 17] [--W 17]
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from efi.configs import AgentConfig, Ablations, EnvConfig
from efi.envs import ForageWorld
from efi.agents.baselines import make_baseline, run_baseline_episode, train_tabular_q
from efi.evaluation import run_experiment


def eval_baseline(name, env_cfg, episodes, seeds, base_seed, q_train_episodes):
    agent = make_baseline(name, seed=base_seed, win=env_cfg.win)
    curve = None
    if getattr(agent, "trains", False):
        def env_factory(ep):
            return ForageWorld(EnvConfig(**{**asdict(env_cfg), "seed": 100_000 + ep}))
        print(f"[baselines] training {name} for {q_train_episodes} episodes...")
        curve = train_tabular_q(env_factory, agent, q_train_episodes)
    rows = []
    for s in range(seeds):
        env = ForageWorld(EnvConfig(**{**asdict(env_cfg), "seed": base_seed + s}))
        for _ in range(episodes):
            rows.append(run_baseline_episode(env, agent))
    return {
        "agent": name,
        "mean_return": float(np.mean([r["return"] for r in rows])),
        "std_return": float(np.std([r["return"] for r in rows])),
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "mean_steps": float(np.mean([r["steps"] for r in rows])),
        "training_curve": curve,
        "n": len(rows),
    }


def eval_efi(env_cfg, episodes, seeds, base_seed):
    agent_cfg = AgentConfig(valA_init=1.0, seed=base_seed)
    res = run_experiment(env_cfg, agent_cfg, None, Ablations(schema=0),
                         episodes=episodes, seeds=seeds, base_seed=base_seed,
                         use_controller=True)
    nA = env_cfg.n_targets_A
    return {
        "agent": "efi",
        "mean_return": res.mean_return,
        "std_return": res.std_return,
        "success_rate": float(np.mean(
            [1.0 if m.targets_collected.get("A", 0) >= nA else 0.0 for m in res.metrics])),
        "mean_steps": res.mean_steps,
        "training_curve": None,  # EFI trains for zero episodes -- the point
        "n": len(res.metrics),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--q-train-episodes", type=int, default=2000)
    ap.add_argument("--H", type=int, default=17)
    ap.add_argument("--W", type=int, default=17)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env_cfg = EnvConfig(H=args.H, W=args.W)

    results = []
    for name in ["random", "greedy", "astar", "q"]:
        r = eval_baseline(name, env_cfg, args.episodes, args.seeds, args.seed,
                          args.q_train_episodes)
        results.append(r)
        print(f"[baselines] {name}: return={r['mean_return']:+.3f} "
              f"success={r['success_rate']:.1%}")
    r = eval_efi(env_cfg, args.episodes, args.seeds, args.seed)
    results.append(r)
    print(f"[baselines] efi: return={r['mean_return']:+.3f} "
          f"success={r['success_rate']:.1%}")

    by = {r["agent"]: r for r in results}
    lo, hi = by["random"]["mean_return"], by["astar"]["mean_return"]
    for r in results:
        r["normalized_score"] = ((r["mean_return"] - lo) / (hi - lo)
                                 if hi > lo else float("nan"))

    out_dir = Path("docs/assets/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"env": asdict(env_cfg),
               "protocol": {"episodes": args.episodes, "seeds": args.seeds,
                            "q_train_episodes": args.q_train_episodes,
                            "note": "identical env seed lists across agents; "
                                    "normalized = (X - random)/(astar - random); "
                                    "astar is a nearest-target greedy-tour ceiling "
                                    "approximation, not the optimal TSP"},
               "results": results}
    with open(out_dir / "baselines.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[baselines] saved to {out_dir / 'baselines.json'}\n")

    print("| Agent | Mean return | Success | Normalized score | Training episodes |")
    print("|---|---|---|---|---|")
    order = ["random", "greedy", "q", "efi", "astar"]
    label = {"random": "Random walk", "greedy": "Greedy-visible",
             "q": f"Tabular Q ({args.q_train_episodes} eps)",
             "efi": "**EFI (0 training)**", "astar": "A* oracle (ceiling)"}
    train_eps = {"random": 0, "greedy": 0, "q": args.q_train_episodes,
                 "efi": 0, "astar": 0}
    for name in order:
        r = by[name]
        print(f"| {label[name]} | {r['mean_return']:+.3f} ± {r['std_return']:.3f} "
              f"| {r['success_rate']:.1%} | {r['normalized_score']:.2f} "
              f"| {train_eps[name]} |")


if __name__ == "__main__":
    main()
