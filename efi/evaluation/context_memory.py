"""EFI-02 common physical streams and autonomous continuous lifetimes."""

from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import gzip
import hashlib
import subprocess
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from ..agents.context_memory import ContextFieldController, ContextMemoryConfig
from ..agents.interaction_controller import InteractionFieldController
from ..agents.interaction_schema import ROTATE, UNROTATE, heading
from ..configs.interaction_config import InteractionConfig
from ..core.anticipation import MOTIONS
from ..envs.context_world import ContextWorld, ContextWorldConfig
from .command_contact import append_frame, recording, score

MODES = ("bank", "fast", "slow", "no_reuse", "no_history")


def make_agent(seed, mode, horizon=2, capacity=3):
    if mode not in MODES:
        raise ValueError("unknown context-memory control")
    if mode in ("fast", "slow"):
        return InteractionFieldController(
            InteractionConfig(horizon=horizon, retention=0.95 if mode == "fast" else 0.995), seed
        )
    return ContextFieldController(
        InteractionConfig(horizon=horizon), seed, ContextMemoryConfig(capacity=capacity), mode
    )


class SourceActor:
    """External acquisition curriculum using ONLY the public local patch.

    Its local shortest detour is evaluator machinery, not agent cognition.
    All executed repositioning steps count. It never receives phase/law IDs.
    """

    def __init__(self, seed):
        self.rng = np.random.RandomState(seed + 17)
        self.queue = deque()
        self.blocked = 0

    def command(self, observation, previous=None):
        if previous is not None:
            self.blocked = self.blocked + 1 if previous["bump"] else 0
        patch = np.asarray(observation).reshape(5, 5, 5)
        objects = np.argwhere(patch[4] > 0.5)
        if self.queue:
            return self.queue.popleft()
        if len(objects) != 1:
            return int(self.rng.randint(5))
        obj = tuple(objects[0])
        facing = heading((2, 2), obj)
        if facing is not None:
            if self.blocked >= 2:
                paths = deque([((2, 2), [])])
                seen = {(2, 2)}
                while paths:
                    body, path = paths.popleft()
                    if body != (2, 2) and heading(body, obj) is not None:
                        self.queue.extend(path)
                        self.blocked = 0
                        return self.queue.popleft()
                    near = heading(body, obj)
                    order = self.rng.permutation(4)
                    for a in order:
                        if near is not None and a in UNROTATE[near, [2, 3]]:
                            continue  # The supplied actuator anchors these moves.
                        dy, dx = MOTIONS[a]
                        nxt = (body[0] + dy, body[1] + dx)
                        if (
                            0 <= nxt[0] < 5
                            and 0 <= nxt[1] < 5
                            and nxt != obj
                            and patch[0, nxt[0], nxt[1]] < 0.5
                            and nxt not in seen
                        ):
                            seen.add(nxt)
                            paths.append((nxt, path + [int(a)]))
            canonical = int(self.rng.choice((0, 2, 3), p=(0.1, 0.45, 0.45)))
            return int(UNROTATE[facing, canonical])
        goals = np.argwhere(patch[1] > 0)
        target = tuple(goals[0]) if len(goals) else obj
        candidates = []
        for a, (dy, dx) in enumerate(MOTIONS[:4]):
            nxt = (2 + dy, 2 + dx)
            if patch[0, nxt[0], nxt[1]] < 0.5 and nxt != obj:
                candidates.append((abs(nxt[0] - target[0]) + abs(nxt[1] - target[1]), a))
        if not candidates:
            return 4
        best = min(d for d, _ in candidates)
        return int(self.rng.choice([a for d, a in candidates if d == best]))


def scored(experience, info):
    result = score(experience, info)
    if result is None:
        return None
    b = int(ROTATE[experience.heading, MOTIONS.index(info["displacement"])])
    o = int(ROTATE[experience.heading, MOTIONS.index(info["object_displacement"])])
    result["wrong"] = int(np.argmax(experience.probabilities) != 5 * b + o)
    return result


def memory_info(agent):
    schema = agent.schema
    return {
        "observed": schema.observed,
        "weights": schema.weights.tolist() if hasattr(schema, "weights") else [1.0],
        "active": int(schema.active.sum()) if hasattr(schema, "active") else 1,
        "admissions": getattr(schema, "admissions", 0),
        "evictions": getattr(schema, "evictions", 0),
        "event": getattr(schema, "last_event", None),
    }


def common_stream(seed, delay=0, modes=MODES, capacity=3, conditions=3, stress=False):
    env = ContextWorld(
        ContextWorldConfig(seed=seed, delay=delay, conditions=conditions, stress=stress)
    )
    observation = env.reset()
    actor = SourceActor(seed)
    agents = {m: make_agent(seed, m, 1, capacity) for m in modes}
    for agent in agents.values():
        agent.observe(observation)
    rows = []
    previous = None
    contact_index = {}
    identifiable_index = {}
    for tick in range(env.max_steps):
        action = actor.command(observation, previous)
        pending = {}
        for mode, agent in agents.items():
            agent.think()
            agent.select_action(action)
            pending[mode] = agent.pending
        observation, reward, _, info = env.step(action)
        first_experience = next(iter(pending.values()))
        scores, memory = {}, {}
        for mode, agent in agents.items():
            scores[mode] = scored(pending[mode], info)
            agent.after_env_step(info["displacement"])
            agent.observe(observation)
            if scores[mode] is not None:
                assert np.isclose(scores[mode]["joint_loss"], agent.last_loss)
            memory[mode] = memory_info(agent)
        phase = info["phase"]
        index = contact_index.get(phase, 0)
        identified = identifiable_index.get(phase, 0)
        if info["contact"]:
            contact_index[phase] = index + 1
            if not info["neutral"]:
                identifiable_index[phase] = identified + 1
        rows.append(
            {
                "seed": seed,
                "delay": delay,
                "tick": tick,
                "action": action,
                "reward": reward,
                "contact_index": index if info["contact"] else None,
                "informative_index": (
                    identified if info["contact"] and not info["neutral"] else None
                ),
                "scores": scores,
                "memory": memory,
                "experience": asdict(first_experience) if first_experience else None,
                "observed_object_after": next(iter(agents.values())).occupant,
                **info,
            }
        )
        previous = {"bump": info["bump"]}
    return {
        "seed": seed,
        "delay": delay,
        "config": asdict(env.cfg),
        "order": env.order,
        "lengths": env.lengths,
        "rows": rows,
        "final_memory": {m: memory_info(a) for m, a in agents.items()},
    }


def behavior_stream(seed, mode="bank", delay=0, capture=False):
    env = ContextWorld(ContextWorldConfig(seed=seed, delay=delay))
    agent = make_agent(seed, mode)
    agent.observe(env.reset())
    demo = recording() if capture else None
    if demo is not None:
        demo["title"] = "EFI-02 · recognizing a returning response"
        demo["guide"].update(
            {
                "title": "A familiar response comes back",
                "description": "One uninterrupted lifetime with changing hidden "
                "actuator responses. "
                "The body and block never reset at changes. Clear the blue block "
                "and enter the green "
                "goal; new goal paint then appears beneath the block. Watch actual "
                "feedback and the "
                "weights of the fast model and retained alternatives.",
                "note": "Complete first held-out bank lifetime, selected before its outcome. "
                "Phase/response captions are evaluator truth and never enter the controller. "
                "This tests recognition and reuse, not prediction of a condition sequence.",
            }
        )
        demo["presentation"]["fps"] = 4
    rows, total = [], 0.0
    previous, reward = None, 0.0
    for tick in range(env.max_steps):
        started = perf_counter()
        agent.think()
        action = agent.select_action()
        elapsed = perf_counter() - started
        pending = agent.pending
        policy = agent.policy.tolist()
        phase = int(np.searchsorted(env.ends, tick, side="right"))
        caption = (
            f"Condition {env.order[phase] + 1} · segment {phase + 1} · "
            f"{env.collections} collections"
        )
        # One scene throughout, so path and telemetry remain continuous.
        append_frame(demo, env, agent, 0, caption, action, previous, reward, total)
        if demo is not None:
            if tick == 0:
                demo["guide"]["chapters"].clear()
                demo["frames"][-1]["info"]["narration"][
                    "feedback"
                ] = "Lifetime begins with empty evidence."
            if tick == 0 or tick == int(env.ends[phase - 1]):
                demo["guide"]["chapters"].append(
                    {"frame": tick, "label": f"{tick}: segment {phase + 1}"}
                )
            info_frame = demo["frames"][-1]["info"]
            info_frame["context_memory"] = memory_info(agent)
            info_frame["narration"]["learning"] += " Weights (fast, archives): " + ", ".join(
                f"{w:.0%}" for w in agent.schema.prior_weights()
            )
        observation, reward, _, info = env.step(action)
        prediction = scored(pending, info)
        started = perf_counter()
        agent.after_env_step(info["displacement"])
        agent.observe(observation)
        latency = 1000 * (elapsed + perf_counter() - started)
        total += reward
        rows.append(
            {
                "seed": seed,
                "mode": mode,
                "delay": delay,
                "tick": tick,
                "action": action,
                "policy": policy,
                "prediction": prediction,
                "latency_ms": latency,
                "reward": reward,
                "memory": memory_info(agent),
                **info,
            }
        )
        previous = info
    append_frame(
        demo, env, agent, 0, caption, info=previous, reward=reward, total=total, terminal=True
    )
    return {
        "seed": seed,
        "mode": mode,
        "delay": delay,
        "collections": env.collections,
        "return": total,
        "steps": env.max_steps,
        "order": env.order,
        "lengths": env.lengths,
        "rows": rows,
    }, demo


def paired(values):
    x = np.asarray(values)
    draws = np.random.RandomState(29).choice(x, (10000, len(x))).mean(axis=1)
    return {
        "mean": float(x.mean()),
        "bootstrap_95": np.percentile(draws, [2.5, 97.5]).tolist(),
        "seed_differences": x.tolist(),
    }


def summarize(streams, behavior):
    summaries = {}
    for delay in (0, 3):
        selected = [s for s in streams if s["delay"] == delay]
        if not selected:
            continue
        groups = {}
        for name, select in (
            ("return", lambda r: r["returning"]),
            ("new", lambda r: not r["returning"] and r["phase"] > 0),
            ("cold", lambda r: r["phase"] == 0),
            ("return_post_neutral", lambda r: r["returning"]),
            ("new_post_neutral", lambda r: not r["returning"] and r["phase"] > 0),
        ):
            per_seed = {}
            for stream in selected:
                rows = [
                    r
                    for r in stream["rows"]
                    if r["contact"]
                    and (
                        r["informative_index"] is not None and r["informative_index"] < 8
                        if name.endswith("post_neutral")
                        else r["contact_index"] < 8
                    )
                    and select(r)
                ]
                per_seed[stream["seed"]] = {
                    m: {
                        k: float(np.mean([r["scores"][m][k] for r in rows]))
                        for k in ("wrong", "joint_loss")
                    }
                    for m in MODES
                }
            groups[name] = {
                "modes": {
                    m: {
                        k: float(np.mean([v[m][k] for v in per_seed.values()]))
                        for k in ("wrong", "joint_loss")
                    }
                    for m in MODES
                },
                "paired": {
                    m: {
                        k: paired([v[m][k] - v["bank"][k] for v in per_seed.values()])
                        for k in ("wrong", "joint_loss")
                    }
                    for m in MODES[1:]
                },
                "per_seed": per_seed,
            }
        coverage = {
            s["seed"]: [
                sum(r["contact"] for r in s["rows"] if r["phase"] == p)
                for p in range(len(s["order"]))
            ]
            for s in selected
        }
        groups["coverage"] = coverage
        summaries[str(delay)] = groups
    behavioral = {}
    for delay in sorted({r["delay"] for r in behavior}):
        rows = [r for r in behavior if r["delay"] == delay]
        values = {m: {r["seed"]: r for r in rows if r["mode"] == m} for m in MODES}
        seed_ids = sorted(values["bank"])
        if any(sorted(v) != seed_ids for v in values.values()):
            raise ValueError("behavior requires matched seed coverage across all modes")
        behavioral[str(delay)] = {
            "modes": {
                m: {
                    "collection_rate": float(
                        np.mean([r["collections"] / r["steps"] for r in values[m].values()])
                    ),
                    "return": float(np.mean([r["return"] for r in values[m].values()])),
                    "contacts": sum(
                        sum(t["contact"] for t in r["rows"]) for r in values[m].values()
                    ),
                    "stalls": sum(sum(t["bump"] for t in r["rows"]) for r in values[m].values()),
                    "steps": sum(r["steps"] for r in values[m].values()),
                    "collections": sum(r["collections"] for r in values[m].values()),
                }
                for m in MODES
            },
            "bank_minus_control": {
                m: paired(
                    [
                        values["bank"][seed]["collections"] / values["bank"][seed]["steps"]
                        - values[m][seed]["collections"] / values[m][seed]["steps"]
                        for seed in seed_ids
                    ]
                )
                for m in MODES[1:]
            },
        }
    primary = summaries["0"]
    ret, new = primary["return"], primary["new"]
    reduction = ret["paired"]["fast"]["wrong"]["mean"] / max(ret["modes"]["fast"]["wrong"], 1e-12)
    gates = {
        "return_relative_reduction": reduction >= 0.25,
        "return_vs_fast": ret["paired"]["fast"]["wrong"]["bootstrap_95"][0] > 0,
        "return_vs_no_reuse": ret["paired"]["no_reuse"]["wrong"]["bootstrap_95"][0] > 0,
        "return_log_loss": ret["paired"]["fast"]["joint_loss"]["bootstrap_95"][0] > 0,
        "new_errors": -new["paired"]["no_reuse"]["wrong"]["bootstrap_95"][0] <= 0.05,
        "new_loss": -new["paired"]["no_reuse"]["joint_loss"]["bootstrap_95"][0] <= 0.25,
        "coverage": all(n >= 8 for counts in primary["coverage"].values() for n in counts),
    }
    gates["behavior"] = (
        behavioral["0"]["bank_minus_control"]["fast"]["bootstrap_95"][0] >= -0.02
        if "0" in behavioral
        else None
    )
    return {
        "common": summaries,
        "behavior": behavioral,
        "relative_error_reduction": reduction,
        "research_gates": gates,
    }


def _seed_experiment(job):
    seed, delays, behavior, capture = job
    streams, behaviors, demo = [], [], None
    for delay in delays:
        streams.append(common_stream(seed, delay))
        if behavior:
            for mode in MODES:
                row, recording_data = behavior_stream(
                    seed, mode, delay, capture=capture and delay == 0 and mode == "bank"
                )
                behaviors.append(row)
                if recording_data is not None:
                    demo = recording_data
    return streams, behaviors, demo


def context_experiment(
    seeds=40, base_seed=61000, output=None, delays=(0, 3), behavior=True, progress=False, workers=1
):
    if seeds < 1 or 0 not in delays or workers < 1:
        raise ValueError(
            "positive seeds/workers and the primary immediate-feedback condition required"
        )
    root = Path(__file__).resolve().parents[2]
    protocol = {
        "base_seed": base_seed,
        "seeds": seeds,
        "delays": list(delays),
        "evaluation_workers": workers,
        "memory": asdict(ContextMemoryConfig()),
        "modes": MODES,
        "base_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ("efi/agents", "efi/core", "efi/envs", "efi/evaluation")
            for p in sorted((root / folder).glob("*.py"))
        },
        "sha256": hashlib.sha256((root / "docs/EFI02_PROTOCOL.md").read_bytes()).hexdigest(),
    }
    streams, behaviors, demo = [], [], None
    jobs = [
        (seed, delays, behavior, seed == base_seed) for seed in range(base_seed, base_seed + seeds)
    ]
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        results = pool.map(_seed_experiment, jobs) if pool else map(_seed_experiment, jobs)
        for i, (common, autonomous, recorded) in enumerate(results):
            streams.extend(common)
            behaviors.extend(autonomous)
            if recorded is not None:
                demo = recorded
            if progress:
                print(
                    f"[EFI-02] seed {base_seed + i}: {len(streams)} common streams; "
                    f"{len(behaviors)} lifetimes",
                    flush=True,
                )
    finally:
        if pool:
            pool.shutdown()
    result = {
        "protocol": protocol,
        "summary": summarize(streams, behaviors),
        "common": streams,
        "behavior": behaviors,
    }
    if output:
        path = Path(output)
        path.mkdir(parents=True, exist_ok=True)
        with gzip.open(path / "results.json.gz", "wt") as handle:
            json.dump(result, handle, default=lambda v: v.tolist(), separators=(",", ":"))
        (path / "summary.json").write_text(
            json.dumps({k: result[k] for k in ("protocol", "summary")}, indent=2)
        )
        if demo is not None:
            from ..visualization.html_viewer import create_html_viewer

            (path / "episode.json").write_text(
                json.dumps(demo, default=lambda v: v.tolist(), separators=(",", ":"))
            )
            create_html_viewer(demo, path / "episode.html")
    return result
