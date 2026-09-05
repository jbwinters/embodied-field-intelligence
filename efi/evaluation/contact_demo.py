"""Record a complete, narrated contact stream in the existing EFI viewer."""

import json
from pathlib import Path

import numpy as np

from ..core.anticipation import MOTIONS
from ..envs.contact_arena import ContactArena, REACTIONS
from ..visualization.html_viewer import create_html_viewer
from .interaction import make_agent, snapshot

NAMES = ("up", "down", "left", "right", "wait")
PHASES = ("Contact pushes forward", "Contact now turns left", "Contact now turns right")


def movement(delta):
    return (
        "stayed in place"
        if tuple(delta) == (0, 0)
        else f"moved {NAMES[MOTIONS.index(tuple(delta))]}"
    )


def contact_demo(seed=6, steps=180, arena="islands", output="runs/contact-demo"):
    env = ContactArena(steps, arena)
    agent = make_agent(seed)
    agent.observe(env.reset())
    recording = {
        "title": "EFI · a continuous contact experiment",
        "presentation": {"fps": 2, "loop": False, "show_policy": False, "field_context": True},
        "guide": {
            "title": "Follow the white agent and the blue block",
            "description": f"{steps} uninterrupted moves, starting with empty contact memory. "
            "Enter the green goal beneath the block to collect it; a new goal then appears beneath "
            "the block's new position. Contact first pushes forward, then turns left, then right, "
            "relative to the attempted move. The agent is not told when the response changes.",
            "legend": [
                {"label": "A · agent", "color": "#f2f3ee"},
                {"label": "B · block", "color": "#3987e5"},
                {"label": "Goal beneath block", "color": "#199e70"},
                {"label": "Chosen next move", "color": "#c98500"},
                {"label": "Dashed box · 5×5 sensing", "color": "#899181"},
            ],
            "chapters": [
                {"frame": i * env.phase_length, "label": f"{i * env.phase_length}: {title}"}
                for i, title in enumerate(PHASES)
            ],
            "note": "This agent looks only two moves ahead. When the block leaves its sight, "
            "it samples moves uniformly until it sees it again. This is an illustrative run, "
            "including its stalls; it does not test longer-horizon planning.",
        },
        "frames": [],
        "world_frames": [],
    }
    rows = []
    reward, total = 0.0, 0.0
    feedback = "No move yet. The learned contact table is empty."
    for step in range(steps + 1):
        terminal = step == steps
        if not terminal:
            agent.think()
            action = agent.select_action()
        phase = min(2, step // env.phase_length)
        caption = (
            f"{PHASES[phase]} · {env.collections} goals collected · "
            f"{agent.schema.observed} observed transitions"
        )
        fields, world = snapshot(env, agent, 0, caption, reward, terminal, total)
        visible = agent.occupant is not None
        learning = "No complete contact prediction was scored on the previous move."
        if agent.last_loss is not None:
            probability = float(np.exp(-agent.last_loss))
            learning = (
                f"Before the update: {probability:.1%} probability for this "
                f"{'agent-and-block movement' if agent.last_complete else 'agent movement'}. "
                "A low percentage means the outcome was surprising."
            )
        fields["info"].update(
            {
                "action": None if terminal else action,
                "narration": {
                    "next": (
                        "Run complete."
                        if terminal
                        else f"Next move: {NAMES[action].upper()}. "
                        + (
                            "The block is inside the sensing window."
                            if visible
                            else "The block is out of sight; this move is sampled uniformly."
                        )
                    ),
                    "feedback": feedback,
                    "learning": learning,
                },
                "sensing_radius": 2,
                "collections": env.collections,
                "markers": [
                    {"pos": list(env.body), "text": "A", "color": "#111310"},
                    {"pos": list(env.occupant), "text": "B", "color": "#f2f3ee"},
                ],
            }
        )
        recording["frames"].append(fields)
        recording["world_frames"].append(world)
        if terminal:
            break
        before_body, before_object = env.body, env.occupant
        observation, reward, _, info = env.step(action)
        agent.after_env_step(info["displacement"])
        agent.observe(observation)
        total += reward
        object_delta = tuple(np.subtract(env.occupant, before_object))
        feedback = (
            f"Previous move: agent {movement(info['displacement'])}; "
            f"block {movement(object_delta)}. "
        )
        if info["success"]:
            feedback += "Goal collected (+1); the next goal is now beneath the block."
        elif info["contact"] and info["bump"]:
            feedback += "Contact was blocked."
        elif info["bump"]:
            feedback += "The move hit a wall."
        rows.append(
            {
                "step": step,
                "action": action,
                "rule": env.cfg.rule,
                "body_before": before_body,
                "body_after": env.body,
                "object_before": before_object,
                "object_after": env.occupant,
                "reward": reward,
                "success": info["success"],
                "bump": info["bump"],
                "contact": info["contact"],
                "displacement": info["displacement"],
                "loss": agent.last_loss,
                "complete": agent.last_complete,
                "learned_transitions": agent.schema.observed,
            }
        )
    result = {
        "seed": seed,
        "steps": steps,
        "arena": arena,
        "source_transitions": 0,
        "collections": env.collections,
        "return": total,
        "phase_collections": {
            rule: sum(r["success"] for r in rows if r["rule"] == rule) for rule in REACTIONS
        },
        "observed_transitions": agent.schema.observed,
        "rows": rows,
        "counts": agent.schema.counts.tolist(),
        "note": "Continuous illustration, no resets or forced commands. Goals replenish after "
        "collection. Two-step planning stops at each reward; future replenishment is not modeled. "
        "The response schedule and narration belong to evaluation, never agent inputs.",
    }
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)

    def dump(value):
        return json.dumps(value, default=lambda x: x.tolist(), separators=(",", ":"))

    (directory / "results.json").write_text(dump(result))
    (directory / "episode.json").write_text(dump(recording))
    create_html_viewer(recording, directory / "episode.html")
    return result
