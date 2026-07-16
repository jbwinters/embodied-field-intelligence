#!/usr/bin/env python3
"""Render a world-grid-only animation of an EFI episode to GIF.

No viewer chrome: just the environment (walls, A targets green, B targets
magenta, agent blue) with the agent's path traced -- the clean "what does
it do" hero shot. Colors come from ForageWorld.render_rgb; the path is
drawn full-history in a light wash with the recent segment emphasized.

Usage:
    python scripts/make_grid_demo.py [--H 35] [--W 35] [--nA 12] [--nB 20]
        [--steps 800] [--seed 4] [--every 4] [--cell 14] [--fps 10]
        [--out docs/assets/images/grid_demo.gif]
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from efi.configs import AgentConfig, Ablations, EnvConfig
from efi.envs import ForageWorld
from efi.agents import FieldController, ForageAdapter
from efi.evaluation import run_episode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=35)
    ap.add_argument("--W", type=int, default=35)
    ap.add_argument("--nA", type=int, default=12)
    ap.add_argument("--nB", type=int, default=20)
    ap.add_argument("--p-wall", type=float, default=0.12)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--every", type=int, default=4, help="render every Nth step")
    ap.add_argument("--cell", type=int, default=14, help="pixels per cell")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--tail", type=int, default=40, help="emphasized recent path steps")
    ap.add_argument("--out", default="docs/assets/images/grid_demo.gif")
    args = ap.parse_args()

    env_cfg = EnvConfig(H=args.H, W=args.W, n_targets_A=args.nA,
                        n_targets_B=args.nB, p_wall=args.p_wall,
                        max_steps=args.steps, seed=args.seed)
    env = ForageWorld(env_cfg)
    env.reset()
    cfg = AgentConfig(valA_init=1.0, seed=args.seed)
    agent = FieldController(env, ForageAdapter(env), cfg,
                            Ablations(schema=0), seed=args.seed)
    _, _, m, ep = run_episode(env, agent, None, Ablations(schema=0),
                              record_fields=True)
    worlds = ep["world_frames"]
    positions = [f["info"]["pos"] for f in ep["frames"]]
    print(f"[grid-demo] episode: A={m.targets_collected.get('A')}/{args.nA} "
          f"B={m.targets_collected.get('B')}/{args.nB} "
          f"return={m.total_return:+.2f} steps={m.steps}")

    C = args.cell
    frames = []
    for t in range(0, len(worlds), args.every):
        img = Image.fromarray(np.asarray(worlds[t], dtype=np.uint8), "RGB")
        img = img.resize((args.W * C, args.H * C), Image.NEAREST)
        draw = ImageDraw.Draw(img)

        def center(p):
            return (p[1] * C + C // 2, p[0] * C + C // 2)

        pts = [center(p) for p in positions[: t + 1]]
        if len(pts) > 1:
            tail_start = max(0, len(pts) - args.tail)
            if tail_start > 1:
                draw.line(pts[:tail_start + 1], fill=(185, 192, 205), width=2)
            draw.line(pts[tail_start:], fill=(50, 90, 210), width=3)
        # agent cell outline on top of the path
        y, x = positions[t]
        draw.rectangle([x * C, y * C, (x + 1) * C - 1, (y + 1) * C - 1],
                       outline=(10, 60, 220), width=2)
        frames.append(img)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=int(1000 / args.fps), loop=0, optimize=True)
    print(f"[grid-demo] wrote {out} ({out.stat().st_size / 1e6:.1f} MB, "
          f"{len(frames)} frames, {frames[0].width}px)")


if __name__ == "__main__":
    main()
