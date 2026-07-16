#!/usr/bin/env python3
"""Capture the web viewer into an animated GIF for the README.

Renders the real viewer HTML in headless Chrome at successive #t=N frames
and assembles them with PIL. Requires google-chrome-stable (or chromium)
on PATH.

Usage:
    python scripts/make_viewer_demo.py [--html runs/interactive_latest.html]
        [--out docs/assets/images/viewer_demo.gif]
        [--every 5] [--width 820] [--fps 7]
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image


def find_chrome():
    for name in ("google-chrome-stable", "google-chrome", "chromium",
                 "chromium-browser"):
        if shutil.which(name):
            return name
    raise SystemExit("no Chrome/Chromium found on PATH")


def n_frames(html_path):
    import re
    m = re.search(r'"n":(\d+)', Path(html_path).read_text())
    if not m:
        raise SystemExit("could not find frame count in the HTML payload")
    return int(m.group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", default="runs/interactive_latest.html")
    ap.add_argument("--out", default="docs/assets/images/viewer_demo.gif")
    ap.add_argument("--every", type=int, default=5, help="capture every Nth step")
    ap.add_argument("--width", type=int, default=820, help="output GIF width")
    ap.add_argument("--fps", type=float, default=7.0)
    ap.add_argument("--viewport", default="1280,1080",
                    help="capture window WxH (before downscale)")
    args = ap.parse_args()

    html = Path(args.html).resolve()
    if not html.exists():
        raise SystemExit(f"{html} not found -- run `python cli.py interactive` first")
    chrome = find_chrome()
    total = n_frames(html)
    steps = list(range(0, total, args.every))
    print(f"[demo] {len(steps)} captures from {total} steps via {chrome}")

    frames = []
    with tempfile.TemporaryDirectory() as td:
        for i, t in enumerate(steps):
            png = Path(td) / f"f{i:04d}.png"
            subprocess.run(
                [chrome, "--headless", "--disable-gpu", "--hide-scrollbars",
                 f"--window-size={args.viewport}",
                 f"--screenshot={png}", f"file://{html}#t={t}"],
                check=True, capture_output=True)
            img = Image.open(png).convert("RGB")
            w = args.width
            img = img.resize((w, int(img.height * w / img.width)), Image.LANCZOS)
            frames.append(img)
            if (i + 1) % 10 == 0:
                print(f"[demo] {i + 1}/{len(steps)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=int(1000 / args.fps), loop=0, optimize=True)
    print(f"[demo] wrote {out} ({out.stat().st_size / 1e6:.1f} MB, "
          f"{len(frames)} frames @ {args.fps:g} fps, {frames[0].width}px wide)")


if __name__ == "__main__":
    main()
