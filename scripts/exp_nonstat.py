#!/usr/bin/env python3
"""Non-stationarity benchmark: regret vs a clairvoyant replanner.

Pre-registered hypotheses: docs/EXPERIMENTS_NONSTAT.md (written first).
Outputs: docs/assets/data/nonstat/*.json, docs/assets/images/nonstat_regret.png.

Usage:
    python scripts/exp_nonstat.py [--seeds 3] [--episodes 3]
        [--q-train-episodes 2000] [--steps 1000]
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from efi.configs import AgentConfig, Ablations, EnvConfig
from efi.envs import ForageWorld
from efi.agents import EgocentricFieldController
from efi.agents.baselines import (AStarOracle, GreedyVisibleAgent, TabularQ,
                                  run_baseline_episode, train_tabular_q)
from efi.evaluation.metrics import adaptation_lag, regret_series, regret_slopes


def condition_cfg(name, steps, seed):
    base = dict(H=17, W=17, max_steps=steps, seed=seed)
    if name == "regrow":
        return EnvConfig(**base, p_regrow=0.02)
    if name == "drift":
        return EnvConfig(**base, T_shift=200)
    if name == "swap":
        return EnvConfig(**base, T_swap=400)
    raise ValueError(name)


def shift_times(name, steps):
    if name == "drift":
        return [t for t in range(200, steps, 200)]
    if name == "swap":
        return [400]
    return []


def run_oracle(env_start):
    """Clairvoyant reference on a clone of the same stochastic world."""
    env = env_start.clone()
    oracle = AStarOracle(seed=0)
    oracle.reset()
    rewards = []
    obs = env._obs()
    for _ in range(env.max_steps - env.t):
        a = oracle.act(obs, env=env)
        obs, r, done, _ = env.step(a)
        rewards.append(r)
        if done:
            break
    return rewards


def run_efi(env, seed):
    cfg = AgentConfig(valA_init=1.0, seed=seed)
    agent = EgocentricFieldController(cfg, Ablations(schema=0), win=env.win, seed=seed)
    from efi.core import AffectState, compute_nociception, update_affect
    obs = env.reset()
    agent.reset()
    oracle_rewards = run_oracle(env)  # clone BEFORE the agent acts
    affect = AffectState() if cfg.affect_enabled else None
    rewards, valence_trace, picks = [], [], []
    stuck = 0
    for _ in range(env.max_steps):
        agent.observe(obs)
        agent.think(affect)
        a = agent.select_action()
        obs, r, done, info = env.step(a)
        moved = bool(info.get("moved", False))
        picked = info.get("picked")
        agent.after_env_step(a, moved, picked)
        rewards.append(r)
        picks.append(picked)
        if picked:
            agent.learn_valence(picked, r - env.cfg.step_cost)
        valence_trace.append((agent.valence["A"], agent.valence["B"]))
        stuck = 0 if moved else stuck + 1
        if affect is not None:
            noci = compute_nociception(not moved, min(0, r), 0.0, stuck,
                                       cfg.pain_bump_weight, cfg.pain_reward_weight,
                                       cfg.pain_prox_weight, cfg.pain_stuck_weight)
            affect = update_affect(affect, noci, agent.last_surprise, r,
                                   cfg.affect_rho_v, cfg.affect_rho_a,
                                   cfg.affect_rho_c, cfg.affect_rho_p)
        if done:
            break
    return rewards, oracle_rewards, valence_trace, picks


def run_baseline(env, agent):
    obs = env.reset()
    agent.reset()
    oracle_rewards = run_oracle(env)
    rewards, picks = [], []
    for _ in range(env.max_steps):
        a = agent.act(obs, env=None)
        obs2, r, done, info = env.step(a)
        agent.observe(obs, a, r, obs2, done)
        obs = obs2
        rewards.append(r)
        picks.append(info.get("picked"))
        if done:
            break
    return rewards, oracle_rewards, None, picks


def train_q(steps, q_train_episodes, seed):
    """Train on the STATIC distribution (the pre-shift world)."""
    def env_factory(ep):
        return ForageWorld(EnvConfig(H=17, W=17, max_steps=200, seed=200_000 + ep))
    agent = TabularQ(seed=seed, eps_start=0.3)
    train_tabular_q(env_factory, agent, q_train_episodes)
    return agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--q-train-episodes", type=int, default=2000)
    args = ap.parse_args()

    out_dir = Path("docs/assets/data/nonstat")
    out_dir.mkdir(parents=True, exist_ok=True)

    q_master = train_q(args.steps, args.q_train_episodes, seed=0)
    print(f"[nonstat] tabular-Q trained: {len(q_master.Q)} states")

    conditions = ["regrow", "drift", "swap"]
    contenders = ["efi", "q_frozen", "q_online", "greedy"]
    results = {c: {a: [] for a in contenders} for c in conditions}

    for cond in conditions:
        for seed in range(args.seeds):
            for ep in range(args.episodes):
                env_seed = 1000 * seed + ep
                for name in contenders:
                    env = ForageWorld(condition_cfg(cond, args.steps, env_seed))
                    if name == "efi":
                        rewards, oracle, valences, picks = run_efi(env, seed)
                    else:
                        if name == "q_frozen":
                            agent = TabularQ(seed=seed)
                            agent.Q = q_master.Q
                            agent.freeze()
                        elif name == "q_online":
                            agent = TabularQ(seed=seed, eps_start=0.05)
                            agent.Q = dict(q_master.Q)  # copy: keeps learning
                            agent.eps = 0.05
                        else:
                            agent = GreedyVisibleAgent(seed=seed)
                        rewards, oracle, valences, picks = run_baseline(env, agent)

                    reg = regret_series(rewards, oracle)
                    sts = shift_times(cond, args.steps)
                    lags = adaptation_lag(rewards, sts) if sts else []
                    entry = {
                        "seed": seed, "episode": ep,
                        "return": float(np.sum(rewards)),
                        "oracle_return": float(np.sum(oracle)),
                        "final_regret": float(reg[-1]),
                        "regret_curve": [float(x) for x in reg[::10]],
                        "regret_slopes": regret_slopes(reg),
                        "adaptation_lags": [(-1 if l is None else int(l)) for l in lags],
                        "picks_A": sum(1 for p in picks if p == "A"),
                        "picks_B": sum(1 for p in picks if p == "B"),
                    }
                    if cond == "swap" and valences is not None:
                        entry["valence_trace"] = [(round(a, 3), round(b, 3))
                                                  for a, b in valences[::20]]
                        # picks of newly-aversive channel (A after swap) post-swap
                        entry["post_swap_picks_A"] = sum(
                            1 for t, p in enumerate(picks) if p == "A" and t >= 400)
                        entry["pre_swap_picks_A"] = sum(
                            1 for t, p in enumerate(picks) if p == "A" and t < 400)
                    results[cond][name].append(entry)
            print(f"[nonstat] {cond} seed {seed} done")

    summary = {}
    for cond in conditions:
        summary[cond] = {}
        for name in contenders:
            rows = results[cond][name]
            lags_flat = [l for r in rows for l in r["adaptation_lags"]]
            recovered = [l for l in lags_flat if l >= 0]
            summary[cond][name] = {
                "mean_return": float(np.mean([r["return"] for r in rows])),
                "mean_final_regret": float(np.mean([r["final_regret"] for r in rows])),
                "mean_regret_slope": float(np.mean([np.mean(r["regret_slopes"]) for r in rows])),
                "n_shifts": len(lags_flat),
                "n_never_recovered": sum(1 for l in lags_flat if l < 0),
                "mean_lag_when_recovered": (float(np.mean(recovered)) if recovered else None),
                "picks_A": float(np.mean([r["picks_A"] for r in rows])),
                "picks_B": float(np.mean([r["picks_B"] for r in rows])),
            }
            if cond == "swap" and name == "efi":
                summary[cond][name]["post_swap_picks_A"] = float(
                    np.mean([r["post_swap_picks_A"] for r in rows]))
                summary[cond][name]["pre_swap_picks_A"] = float(
                    np.mean([r["pre_swap_picks_A"] for r in rows]))

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f)
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"protocol": vars(args), "summary": summary}, f, indent=2)
    print(json.dumps(summary, indent=2))

    # Regret plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
        for ax, cond in zip(axes, conditions):
            for name in contenders:
                curves = [r["regret_curve"] for r in results[cond][name]]
                n = min(len(c) for c in curves)
                mean_curve = np.mean([c[:n] for c in curves], axis=0)
                ax.plot(np.arange(n) * 10, mean_curve, label=name)
            for s in shift_times(cond, args.steps):
                ax.axvline(s, color="gray", ls=":", lw=0.8)
            ax.set_title(cond)
            ax.set_xlabel("step")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("cumulative regret vs clairvoyant")
        axes[0].legend()
        fig.tight_layout()
        Path("docs/assets/images").mkdir(parents=True, exist_ok=True)
        fig.savefig("docs/assets/images/nonstat_regret.png", dpi=120)
        print("[nonstat] plot saved to docs/assets/images/nonstat_regret.png")
    except ImportError:
        print("[nonstat] matplotlib unavailable; skipping plot")


if __name__ == "__main__":
    main()
