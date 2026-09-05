"""Paired online-learning, transfer, and changed-rule crossing experiment.

Each contender gets the same initial worlds, five actions, observation
window, horizon, and online experience allowance. Worlds have deterministic
hazard trajectories independent of agent actions. Parameters persist within
a seed across three phases; each seed starts with an untrained model.
"""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from ..agents.anticipatory_controller import AnticipatoryFieldController
from ..configs import AgentConfig, Ablations
from ..configs.anticipation_config import AnticipationConfig
from ..envs.crossing_world import CrossingConfig, CrossingWorld


def make_crossing_agent(mode="learned", seed=0, horizon=4):
    cfg = AgentConfig(
        map_size=31,
        valA_init=1.0,
        affect_enabled=False,
        membrane_enabled=False,
        pose_correction=False,
    )
    return AnticipatoryFieldController(
        cfg,
        Ablations(trail=0, corner=0),
        AnticipationConfig(horizon=horizon, forecast_mode="learned" if mode == "frozen" else mode),
        win=5,
        seed=seed,
    )


def run_crossing_episode(env, agent, record=False):
    obs = env.reset()
    agent.reset()
    origin = agent.pose
    start = env.y, env.x
    total = 0.0
    waits = 0
    frames = []
    losses = []
    updates_before = agent.motion.transitions

    def world_view(field):
        # Measurement only: align the internal map for the human viewer.
        y0, x0 = origin[0] - start[0], origin[1] - start[1]
        return field[y0 : y0 + env.H, x0 : x0 + env.W].round(4).tolist()

    for t in range(env.max_steps):
        agent.observe(obs)
        agent.think()
        action = agent.select_action()
        if agent.motion.last_loss is not None:
            losses.append(agent.motion.last_loss)
        if record:
            frames.append(
                {
                    "step": t,
                    "position": [env.y, env.x],
                    "hazard": list(env.hazard),
                    "goal": list(env.goal),
                    "walls": env.walls.astype(int).tolist(),
                    "known_walls": world_view(agent.known_walls.astype(np.float32)),
                    "seen": world_view(agent.seen.astype(np.float32)),
                    "forecast": [world_view(f) for f in agent.forecasts],
                    "policy": agent.policy.tolist(),
                    "action": action,
                    "transitions": agent.motion.transitions,
                }
            )
        obs, reward, done, info = env.step(action)
        # The closed-box controller gets observations and proprioception;
        # collision/success labels and world coordinates stay in evaluation.
        agent.after_env_step(action, info["moved"], info["picked"])
        total += reward
        waits += int(info["wait"])
        if done:
            break
    row = {
        "return": float(total),
        "success": info["success"],
        "collision": info["collision"],
        "timeout": not done or (not info["success"] and not info["collision"]),
        "steps": t + 1,
        "waits": waits,
        "learned_transitions": agent.motion.transitions - updates_before,
        "prediction_log_loss": float(np.mean(losses)) if losses else None,
    }
    return row, frames


def crossing_experiment(seeds=12, episodes=20, base_seed=1000, horizon=4, output=None, record=True):
    if seeds < 1 or episodes < 1:
        raise ValueError("seeds and episodes must be positive")
    modes = ("learned", "static", "unlearned", "frozen")
    phases = (
        ("acquire", 9, 9, "continue"),
        ("transfer", 11, 13, "continue"),
        ("reverse", 11, 13, "reverse"),
    )
    rows = []
    demo = None
    for s in range(seeds):
        seed = base_seed + s
        agents = {m: make_crossing_agent(m, seed, horizon) for m in modes}
        for phase, H, W, rule in phases:
            for episode in range(episodes):
                cfg = CrossingConfig(
                    H=H, W=W, seed=seed * 10000 + episode, rule=rule, rotate=seed % 4
                )
                for mode, agent in agents.items():
                    if mode == "frozen" and phase == "reverse":
                        agent.anticipation.learn_motion = False
                    want_record = (
                        record and mode == "learned" and phase == "transfer" and demo is None
                    )
                    row, frames = run_crossing_episode(CrossingWorld(cfg), agent, want_record)
                    row.update(seed=seed, phase=phase, episode=episode, mode=mode)
                    rows.append(row)
                    if want_record and row["success"] and row["waits"] > 0:
                        demo = {"config": asdict(cfg), "result": row.copy(), "frames": frames}

    summary = {}
    for phase, *_ in phases:
        summary[phase] = {}
        for mode in modes:
            group = [r for r in rows if r["phase"] == phase and r["mode"] == mode]
            summary[phase][mode] = {
                k: float(np.mean([r[k] for r in group]))
                for k in ("return", "success", "collision", "timeout", "steps", "waits")
            }
            summary[phase][mode]["n"] = len(group)
        # Paired seed differences, not trials treated as independent seeds.
        for comparison in ("static", "unlearned", "frozen"):
            differences = []
            for s in range(seeds):
                means = {}
                for m in ("learned", comparison):
                    means[m] = np.mean(
                        [
                            r["success"]
                            for r in rows
                            if r["phase"] == phase and r["seed"] == base_seed + s and r["mode"] == m
                        ]
                    )
                differences.append(float(means["learned"] - means[comparison]))
            summary[phase]["paired_success_vs_" + comparison] = differences
    payload = {
        "protocol": {
            "seeds": seeds,
            "episodes_per_phase": episodes,
            "base_seed": base_seed,
            "horizon": horizon,
            "modes": list(modes),
            "initial_training_episodes": 0,
            "phases": [p[0] for p in phases],
            "note": "Learning continues across trials; transfer and reverse reuse acquired rules. "
            "All trials, including acquisition and timeouts, are reported. "
            "Motion law changes between phases; geometry changes at transfer.",
            "frozen_control": (
                "Identical to learned until reversal; then only motion-rule updates stop."
            ),
        },
        "summary": summary,
        "rows": rows,
    }
    if output is not None:
        out = Path(output)
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
        if demo is not None:
            from ..visualization.crossing_viewer import save_crossing_viewer

            (out / "episode.json").write_text(json.dumps(demo) + "\n")
            save_crossing_viewer(demo, out / "episode.html")
    return payload
