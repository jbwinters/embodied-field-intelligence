"""Reproducible acquisition, common-stream scoring, and contact transfer.

Evaluation may see world truth for recording and labels. Agent inputs remain
only observation windows and proprioceptive displacement. All attempted
contacts and failures remain in the data.
"""

from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from ..agents.interaction_controller import InteractionFieldController
from ..configs.interaction_config import InteractionConfig
from ..core.anticipation import MOTIONS
from ..envs.interaction_world import InteractionWorld, InteractionWorldConfig

MODES = ("online", "frozen", "empty", "shuffled", "passive", "one_step", "tabular")
LAWS = ("push", "left", "right")
LAYOUTS = ("west", "north", "detour")
CONTEXTS = tuple(code for code in range(16) if not code & 2)


def make_agent(seed=0, mode="online", source=False):
    if mode not in MODES:
        raise ValueError("unknown interaction control")
    cfg = InteractionConfig(
        horizon=1 if source or mode == "one_step" else 2,
        learn=mode == "online",
        action_conditioned=mode != "passive",
    )
    return InteractionFieldController(cfg, seed, reference=mode == "tabular")


def snapshot(env, agent, scene, caption, reward=0.0, terminal=False, episode_return=0.0):
    """Adapt real agent fields to the ORIGINAL viewer's payload contract."""
    size = 13
    pad = (size - env.H) // 2
    offset = agent.cfg.map_size // 2 - env.H // 2

    def place(field):
        out = np.zeros((size, size), dtype=np.float32)
        out[pad : pad + env.H, pad : pad + env.W] = field
        return out

    def local(channel):
        return place(agent.memory[offset : offset + env.H, offset : offset + env.W, channel])

    fields = {"Goal": local(1), "Object": local(4)}
    fields["Walls"] = local(0) > 0.5
    for key in ("ActionValue", "Unresolved", "ObjectFuture", "BodyNext"):
        fields[key] = np.zeros((size, size), dtype=np.float32)
    pos = np.asarray(env.body) + pad
    if not terminal and hasattr(agent.field, "first"):
        p, unknown, bnext, onext, *_ = agent.field.first
        for action, (dy, dx) in enumerate(MOTIONS):
            fields["ActionValue"][tuple(pos + (dy, dx))] = agent.action_values[action]
            fields["Unresolved"][tuple(pos + (dy, dx))] = unknown[action]
        for a in range(5):
            for effect in range(25):
                mass = p[a, effect] * agent.policy[a]
                for key, nxt in (("BodyNext", bnext),):
                    y, x = pos + nxt[a, effect] - 4
                    if 0 <= y < size and 0 <= x < size:
                        fields[key][y, x] += mass
        coordinates, forecast_mass = agent.field.object_forecast(agent.policy)
        coordinates = coordinates + pos - 4
        valid = ((coordinates >= 0) & (coordinates < size)).all(axis=1)
        np.add.at(
            fields["ObjectFuture"],
            (coordinates[valid, 0], coordinates[valid, 1]),
            forecast_mass[valid],
        )
    rgb = np.full((size, size, 3), (18, 19, 18), dtype=np.uint8)
    room = np.full((env.H, env.W, 3), (38, 42, 37), dtype=np.uint8)
    room[env.walls] = (111, 119, 106)
    room[env.goals > 0] = (25, 158, 112)
    room[env.hazards > 0] = (230, 103, 103)
    room[env.occupant] = (57, 135, 229)
    room[env.body] = (242, 243, 238)
    rgb[pad : pad + env.H, pad : pad + env.W] = room
    goals = (np.argwhere(env.goals > 0) + pad).tolist()
    fields["info"] = {
        "pos": pos.tolist(),
        "scene": int(scene),
        "caption": caption,
        "policy": None if terminal else agent.policy.tolist(),
        "value_bounds": None if terminal else agent.field.value_bounds.tolist(),
        "lam": agent.cfg.temperature,
        "reward": float(reward),
        "learned_transitions": float(agent.schema.counts.sum()),
        "prediction_loss": agent.last_loss,
        "goal_markers": goals,
        "step": env.t,
        "return": float(episode_return),
    }
    return fields, rgb


def run_episode(env, agent, record=None, scene=0, caption="", forced=None):
    obs = env.reset()
    agent.reset()
    agent.observe(obs)
    total, contacts, bumps = 0.0, 0, 0
    reward = 0.0
    losses, latency, terms = [], [], []
    for step in range(env.cfg.max_steps):
        started = perf_counter()
        agent.think()
        action = agent.select_action(forced)
        elapsed = perf_counter() - started
        if record is not None:
            f, world = snapshot(env, agent, scene, caption, reward, episode_return=total)
            record["frames"].append(f)
            record["world_frames"].append(world)
        obs, reward, done, info = env.step(action)
        started = perf_counter()
        agent.after_env_step(info["displacement"])
        agent.observe(obs)  # Includes learning from the final physical transition.
        latency.append(1000 * (elapsed + perf_counter() - started))
        terms.append(agent.field.outcome_terms)
        if agent.last_loss is not None:
            losses.append(agent.last_loss)
        total += reward
        contacts += int(info["contact"])
        bumps += int(info["bump"])
        if done:
            break
    if record is not None:
        status = "collected" if info["success"] else "attempt complete"
        f, world = snapshot(env, agent, scene, caption + " · " + status, reward, True, total)
        record["frames"].append(f)
        record["world_frames"].append(world)
    return {
        "success": bool(info["success"]),
        "return": float(total),
        "steps": step + 1,
        "contacts": contacts,
        "bumps": bumps,
        "collision": bool(info["collision"]),
        "loss": float(np.mean(losses)) if losses else None,
        "latency_ms": latency,
        "outcome_terms": terms,
    }


def acquire(seed, law, repetitions, capture=None):
    learner = make_agent(seed, source=True)
    empty = make_agent(seed, "empty", source=True)
    rows = []
    rng = np.random.RandomState(seed)
    tasks = [(context, a) for context in CONTEXTS for a in range(5)]
    for rep in range(repetitions):
        for index in rng.permutation(len(tasks)):
            context, action = tasks[index]
            rotation = int(rng.randint(4))
            # Transform the experimental command with the room, while the
            # learner sees only the resulting public primitive command.
            vector = np.asarray(MOTIONS[action])
            for _ in range(rotation):
                vector = np.asarray((-vector[1], vector[0]))
            actual_action = MOTIONS.index(tuple(vector))
            cfg = InteractionWorldConfig(
                rule=law,
                acquisition=True,
                context=context,
                max_steps=1,
                rotate=rotation,
                seed=seed * 1000 + rep * 40 + int(index),
            )
            record = capture if action == 0 and context == 0 else None
            row = run_episode(
                InteractionWorld(cfg),
                learner,
                record,
                len(rows),
                "Acquire · controlled contact · other interventions omitted",
                actual_action,
            )
            baseline = run_episode(InteractionWorld(cfg), empty, forced=actual_action)
            rows.append(
                {
                    "seed": seed,
                    "law": law,
                    "exposure": len(rows),
                    "context": context,
                    "command": action,
                    "loss": row["loss"],
                    "empty_loss": baseline["loss"],
                    "contact": action == 0,
                    "complete": learner.last_complete,
                    "contacts": row["contacts"],
                    "bumps": row["bumps"],
                    "return": row["return"],
                    "steps": row["steps"],
                    "latency_ms": row["latency_ms"][0],
                }
            )
    return learner.schema.counts.copy(), rows


def paired(rows, mode, baseline, metric, seeds):
    differences = []
    for seed in seeds:
        means = [
            np.mean([r[metric] for r in rows if r["seed"] == seed and r["mode"] == m])
            for m in (mode, baseline)
        ]
        differences.append(float(means[0] - means[1]))
    samples = np.random.RandomState(29).choice(differences, (10000, len(seeds)))
    return {
        "mean": float(np.mean(differences)),
        "seed_differences": differences,
        "bootstrap_95": np.percentile(samples.mean(axis=1), [2.5, 97.5]).tolist(),
    }


def interaction_experiment(
    seeds=40, episodes=8, acquisition=2, base_seed=10000, output=None, progress=False
):
    if min(seeds, episodes, acquisition) < 1:
        raise ValueError("positive seeds, episodes and acquisition repetitions required")
    rows, training, models = [], [], []
    demo = {"frames": [], "world_frames": [], "title": "EFI · learned action consequences"}
    for seed in range(base_seed, base_seed + seeds):
        for law in LAWS:
            counts, source = acquire(
                seed, law, acquisition, demo if seed == base_seed and law == "push" else None
            )
            training.extend(source)
            models.append({"seed": seed, "law": law, "counts": counts.tolist()})
            for mode in MODES:
                agent = make_agent(seed, mode)
                if mode != "empty":
                    agent.schema.counts[:] = counts
                    if mode == "shuffled":
                        perm = np.random.RandomState(seed + 31).permutation(25)
                        agent.schema.counts[:] = counts[:, :, perm]
                initial = agent.schema.counts.copy()
                for layout in LAYOUTS:
                    for episode in range(episodes):
                        cfg = InteractionWorldConfig(
                            seed=seed * 1000 + episode,
                            rule=law,
                            layout=layout,
                            rotate=(seed + episode) % 4,
                            size=(9, 11, 13)[episode % 3],
                        )
                        # Capture the first two trials of two named controls,
                        # selected before outcomes, in the existing viewer.
                        record = (
                            demo
                            if (
                                seed == base_seed
                                and law == "push"
                                and mode in ("online", "empty")
                                and layout in ("west", "north")
                                and episode < 2
                            )
                            else None
                        )
                        scene = len(rows) + len(training)
                        caption = "Acquired + online" if mode == "online" else "Empty evidence"
                        caption += f" · {layout} arrangement · trial {episode + 1}"
                        row = run_episode(InteractionWorld(cfg), agent, record, scene, caption)
                        rows.append(
                            dict(
                                row,
                                seed=seed,
                                law=law,
                                mode=mode,
                                layout=layout,
                                episode=episode,
                                learned_transitions=agent.schema.observed,
                            )
                        )
                if mode != "online" and not np.array_equal(initial, agent.schema.counts):
                    raise AssertionError("a frozen control changed its empirical counts")
        if progress:
            print(f"[interaction] seed {seed}: {len(rows)} target trials", flush=True)
    summary = {}
    seed_ids = list(range(base_seed, base_seed + seeds))
    for layout in LAYOUTS:
        group = [r for r in rows if r["layout"] == layout]
        summary[layout] = {}
        for mode in MODES:
            selection = [r for r in group if r["mode"] == mode]
            summary[layout][mode] = {
                key: float(np.mean([r[key] for r in selection]))
                for key in ("success", "return", "steps", "contacts", "bumps", "collision")
            }
            summary[layout][mode]["n"] = len(selection)
        for baseline in MODES[1:]:
            summary[layout]["paired_vs_" + baseline] = {
                key: paired(group, "online", baseline, key, seed_ids)
                for key in ("success", "return")
            }
    all_latency = [v for r in rows if r["mode"] == "online" for v in r["latency_ms"]]
    payload = {
        "protocol": {
            "seeds": seeds,
            "base_seed": base_seed,
            "episodes": episodes,
            "acquisition_repetitions": acquisition,
            "source_exposures_per_agent": acquisition * 40,
            "source_contacts_per_agent": acquisition * 8,
            "modes": list(MODES),
            "laws": list(LAWS),
            "layouts": list(LAYOUTS),
            "config": asdict(InteractionConfig()),
            "development_seeds": list(range(10)),
            "bootstrap": "10000 paired seed resamples, RNG 29, percentile 95%",
            "note": "One fast context table. No multiple-context retention or learned motor "
            "composition claim. Two-step local terminating outcomes. Only online "
            "continues learning on target trials; other controls freeze evidence.",
        },
        "summary": summary,
        "prediction": {
            "prequential_log_loss": float(np.mean([r["loss"] for r in training])),
            "empty_log_loss": float(np.mean([r["empty_loss"] for r in training])),
            "complete_fraction": float(np.mean([r["complete"] for r in training])),
        },
        "resources": {
            "latency_ms_percentiles": dict(
                zip(("p50", "p95", "p99"), np.percentile(all_latency, [50, 95, 99]).tolist())
            ),
            "max_outcome_terms": max(v for r in rows for v in r["outcome_terms"]),
        },
        "rows": rows,
        "training": training,
        "models": models,
    }
    if output:
        directory = Path(output)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "results.json").write_text(json.dumps(payload, separators=(",", ":")))
        (directory / "summary.json").write_text(
            json.dumps(
                {k: v for k, v in payload.items() if k not in ("rows", "training", "models")},
                indent=2,
            )
        )
        from ..visualization.html_viewer import create_html_viewer

        (directory / "episode.json").write_text(
            json.dumps(demo, default=lambda value: value.tolist(), separators=(",", ":"))
        )
        create_html_viewer(demo, directory / "episode.html")
    return payload
