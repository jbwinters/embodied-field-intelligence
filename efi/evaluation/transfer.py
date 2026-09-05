"""Transfer learned hazard dynamics to rewarding moving objects in rooms.

Each seed learns from crossing experience. Only its finite motion table is
copied into fresh target controllers. Spatial memory, target experience and
RNG state are not transferred. Frozen controls isolate reuse from adaptation.
"""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from ..agents.anticipatory_controller import AnticipatoryFieldController
from ..configs import AgentConfig, Ablations
from ..configs.anticipation_config import AnticipationConfig
from ..envs.crossing_world import CrossingConfig, CrossingWorld
from ..envs.interception_world import InterceptionConfig, InterceptionWorld
from .crossing import run_crossing_episode

MODES = (
    "transfer",
    "scratch",
    "static",
    "exact",
    "uncorrected",
    "online",
    "scratch_online",
    "one_step",
)
TASKS = (
    ("room", 9, 13, False, False),
    ("obstacles", 11, 15, True, False),
    ("mixed", 11, 15, False, True),
    ("mixed_obstacles", 11, 15, True, True),
)


def make_transfer_agent(seed=0, mode="transfer", targets=True, horizon=4):
    if mode not in MODES:
        raise ValueError("unknown transfer mode")
    cfg = AgentConfig(
        map_size=31,
        valA_init=1.0,
        affect_enabled=False,
        membrane_enabled=False,
        pose_correction=False,
        lam_base=0.005 if targets else 0.02,
    )
    anticipation = AnticipationConfig(
        horizon=1 if mode == "one_step" else horizon,
        relational_motion=True,
        moving_targets=targets,
        pool_motion=mode != "exact",
        correct_tracks=mode != "uncorrected",
        forecast_mode="static" if mode == "static" else "learned",
        learn_motion=not targets or mode in ("online", "scratch_online"),
    )
    return AnticipatoryFieldController(
        cfg,
        Ablations(trail=0, corner=0),
        anticipation,
        win=5,
        seed=seed,
    )


def run_interception_episode(env, agent, record=False):
    obs = env.reset()
    agent.reset()
    origin, start = agent.pose, (env.y, env.x)
    total, waits, bumps = 0.0, 0, 0
    frames, losses = [], []
    before = agent.target_motion.transitions

    def world_view(field):
        y0, x0 = origin[0] - start[0], origin[1] - start[1]
        return field[y0 : y0 + env.H, x0 : x0 + env.W].round(4).tolist()

    for t in range(env.max_steps):
        agent.observe(obs)
        agent.think()
        action = agent.select_action()
        if agent.target_motion.last_loss is not None:
            losses.append(agent.target_motion.last_loss)
        if record:
            frames.append(
                {
                    "step": t,
                    "position": [env.y, env.x],
                    "target": list(env.target),
                    "hazard": list(env.hazard) if env.hazard is not None else None,
                    "walls": env.walls.astype(int).tolist(),
                    "known_walls": world_view(agent.known_walls.astype(np.float32)),
                    "seen": world_view(agent.seen.astype(np.float32)),
                    "forecast": [world_view(f) for f in agent.target_forecasts],
                    "hazard_forecast": [world_view(f) for f in agent.forecasts],
                    "policy": agent.policy.tolist(),
                    "action": action,
                }
            )
        obs, reward, done, info = env.step(action)
        agent.after_env_step(action, info["moved"], info["picked"])
        total += reward
        waits += int(info["wait"])
        bumps += int(info["bump"])
        if done:
            break
    return {
        "success": info["success"],
        "return": float(total),
        "steps": t + 1,
        "collision": info["collision"],
        "timeout": not info["success"] and not info["collision"],
        "waits": waits,
        "bumps": bumps,
        "learned_transitions": agent.target_motion.transitions - before,
        "prediction_log_loss": float(np.mean(losses)) if losses else None,
    }, frames


def transfer_experiment(
    seeds=12, episodes=12, acquisition=20, base_seed=5000, horizon=4, output=None, record=True
):
    if min(seeds, episodes, acquisition) < 1:
        raise ValueError("positive seed, episode and acquisition counts required")
    rows, training, models = [], [], []
    demo = None
    for seed in range(base_seed, base_seed + seeds):
        source = make_transfer_agent(seed, targets=False, horizon=horizon)
        for episode in range(acquisition):
            cfg = CrossingConfig(seed=seed * 10000 + episode, rotate=seed % 4)
            row, _ = run_crossing_episode(CrossingWorld(cfg), source)
            training.append(dict(row, seed=seed, episode=episode))
        counts = source.motion.counts.copy()
        models.append({"seed": seed, "counts": counts.tolist()})
        for task, H, W, obstacles, hazards in TASKS:
            agents = {mode: make_transfer_agent(seed, mode, horizon=horizon) for mode in MODES}
            for mode, agent in agents.items():
                if mode not in ("scratch", "scratch_online"):
                    agent.motion.counts[:] = counts
            for episode in range(episodes):
                cfg = InterceptionConfig(
                    H=H,
                    W=W,
                    obstacles=obstacles,
                    hazards=hazards,
                    seed=seed * 10000 + 5000 + episode,
                    rotate=seed % 4,
                )
                for mode, agent in agents.items():
                    capture = (
                        record and demo is None and task == "mixed_obstacles" and mode == "transfer"
                    )
                    row, frames = run_interception_episode(InterceptionWorld(cfg), agent, capture)
                    rows.append(dict(row, seed=seed, episode=episode, task=task, mode=mode))
                    if capture:
                        # First trial, selected before its outcome is known.
                        demo = {
                            "kind": "interception",
                            "config": asdict(cfg),
                            "result": rows[-1],
                            "frames": frames,
                        }
            for mode, agent in agents.items():
                if mode not in ("online", "scratch_online"):
                    expected = np.zeros_like(counts) if mode == "scratch" else counts
                    if not np.array_equal(agent.motion.counts, expected):
                        raise AssertionError("frozen transfer changed its learned evidence")
    summary = {}
    for task, *_ in TASKS:
        summary[task] = {}
        for mode in MODES:
            group = [r for r in rows if r["task"] == task and r["mode"] == mode]
            summary[task][mode] = {
                k: float(np.mean([r[k] for r in group]))
                for k in ("success", "return", "steps", "collision", "timeout", "waits", "bumps")
            }
            summary[task][mode]["n"] = len(group)
        for comparison in MODES[1:]:
            paired = {}
            for metric in ("success", "return"):
                differences = []
                for seed in range(base_seed, base_seed + seeds):
                    means = {
                        mode: np.mean(
                            [
                                r[metric]
                                for r in rows
                                if r["task"] == task and r["mode"] == mode and r["seed"] == seed
                            ]
                        )
                        for mode in ("transfer", comparison)
                    }
                    differences.append(float(means["transfer"] - means[comparison]))
                samples = np.random.RandomState(17).choice(differences, (10000, seeds))
                paired[metric] = {
                    "mean": float(np.mean(differences)),
                    "seed_differences": differences,
                    "bootstrap_95": np.percentile(samples.mean(axis=1), [2.5, 97.5]).tolist(),
                }
            summary[task]["paired_vs_" + comparison] = paired
    payload = {
        "protocol": {
            "seeds": seeds,
            "base_seed": base_seed,
            "episodes": episodes,
            "acquisition": acquisition,
            "horizon": horizon,
            "modes": list(MODES),
            "source_temperature": 0.02,
            "target_temperature": 0.005,
            "target_sweeps": 8,
            "target_terminal_weight": 0.5,
            "correction_sweeps": 4,
            "map_size": 31,
            "win": 5,
            "task_steps": 24,
            "source_steps": 60,
            "tasks": [list(task) for task in TASKS],
            "development_seeds": [3000, 3001, 3002, 3003],
            "bootstrap": "10000 paired seed resamples; RNG seed 17; percentile 95%",
            "note": "Hazard avoidance -> moving reward interception. Only counts transfer. "
            "Each task starts from the same acquired counts; scratch variants start empty. "
            "Only online variants learn during transfer. "
            "Source and target motion axes are perpendicular within each seed.",
        },
        "summary": summary,
        "training": training,
        "models": models,
        "rows": rows,
    }
    if output is not None:
        out = Path(output)
        out.mkdir(parents=True, exist_ok=True)
        # Keep raw evidence reviewable: one trial/model per line, with the
        # protocol and summary indented normally. This remains ordinary JSON.
        header = json.dumps(
            {k: v for k, v in payload.items() if k not in ("training", "models", "rows")}, indent=2
        )
        parts = [header[:-2]]
        for key in ("training", "models", "rows"):
            parts.append(',\n"' + key + '": [\n')
            parts.append(",\n".join(json.dumps(row) for row in payload[key]))
            parts.append("\n]")
        parts.append("\n}\n")
        (out / "results.json").write_text("".join(parts))
        if demo is not None:
            from ..visualization.crossing_viewer import save_crossing_viewer

            (out / "episode.json").write_text(json.dumps(demo) + "\n")
            save_crossing_viewer(demo, out / "episode.html")
    return payload
