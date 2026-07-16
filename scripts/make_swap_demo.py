#!/usr/bin/env python3
"""Generate the policy-reversal (reward swap) viewer demo.

Runs 25x25 ForageWorld episodes with regrowing targets and a mid-episode
reward swap (A and B exchange values at T_swap), auditions a few seeds for
a demonstrative episode (pickups on both sides of the swap, valences
actually flipped), and writes the winner to runs/swap_demo.html.

Then: python scripts/make_viewer_demo.py --html runs/swap_demo.html \
          --out docs/assets/images/viewer_swap_demo.gif \
          --every 8 --width 780 --fps 8 --viewport 1280,1250
"""

import argparse

from efi.configs import AgentConfig, Ablations, EnvConfig
from efi.envs import ForageWorld
from efi.agents import FieldController, ForageAdapter
from efi.evaluation import run_episode
from efi.visualization.html_viewer import create_html_viewer


def audition(seed, t_swap, steps):
    env_cfg = EnvConfig(H=25, W=25, n_targets_A=4, n_targets_B=4,
                        max_steps=steps, T_swap=t_swap, p_regrow=0.03, seed=seed)
    env = ForageWorld(env_cfg)
    env.reset()
    cfg = AgentConfig(valA_init=1.0, seed=seed)
    agent = FieldController(env, ForageAdapter(env), cfg, Ablations(schema=0), seed=seed)
    _, _, m, ep = run_episode(env, agent, None, Ablations(schema=0), record_fields=True)

    pre = post = 0
    for f in ep["frames"]:
        r, t = f["info"]["reward"], f["info"]["step"]
        if r > 0.5:  # a positive pickup (A before the swap, B after)
            pre += t < t_swap
            post += t >= t_swap
    wA = ep["frames"][-1]["info"]["valA"]
    wB = ep["frames"][-1]["info"]["valB"]
    flipped = wA < 0 < wB
    score = min(pre, 6) + min(post, 6) + 4 * flipped
    return score, pre, post, flipped, ep, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--t-swap", type=int, default=300)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--out", default="runs/swap_demo.html")
    args = ap.parse_args()

    best = None
    for seed in range(args.seeds):
        score, pre, post, flipped, ep, m = audition(seed, args.t_swap, args.steps)
        print(f"[swap-demo] seed {seed}: +picks pre={pre} post={post} "
              f"flipped={flipped} score={score}")
        if best is None or score > best[0]:
            best = (score, seed, ep, m)

    score, seed, ep, m = best
    print(f"[swap-demo] using seed {seed}")
    path = create_html_viewer(ep, args.out, final_metrics={
        "total_return": m.total_return, "steps": m.steps, "coverage": m.coverage,
        "targets_A": m.targets_collected.get("A"),
        "targets_B": m.targets_collected.get("B")})
    print(f"[swap-demo] viewer: {path}")


if __name__ == "__main__":
    main()
