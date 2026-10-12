# Render the README hero animation: the same map, the same four spawn points,
# two teams side by side.
#
#   left  -- independent frontier explorers, no communication
#   right -- decentralized coordinated frontier assignment + shared maps
#
# Cells the team has not seen yet are drawn as fog; ground-truth walls are only
# revealed once a robot senses them. Each panel freezes when its team finishes
# or deadlocks.
#
# Usage:  python benchmarks/make_demo.py [--seed 7] [--out assets/demo.gif]

from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from environment import Environment
from policy import make_policy
from run_benchmarks import CONFIGS, DEADLOCK_WINDOW

CELL = 14
PAD = 16
HEADER = 58
FOOTER = 20

# Chart chrome (dark surface so the fog reads as "unknown").
SURFACE = (26, 26, 25)
FOG = (44, 44, 42)
FREE = (236, 235, 230)
WALL = (110, 109, 104)
INK = (255, 255, 255)
INK_2 = (195, 194, 183)
TRACK = (56, 56, 53)
GOOD = (12, 163, 12)
CRITICAL = (230, 103, 103)
# Categorical slots 1-4 (dark-mode steps), one per robot.
ROBOT_COLORS = [(57, 135, 229), (217, 89, 38), (25, 158, 112), (201, 133, 0)]


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", name)
    return ImageFont.truetype(path, size)


def simulate(config: str, seed: int, n: int, size: int, density: float, max_ticks: int):
    """Run one episode and return per-tick frames of (team belief, positions, trails)."""
    policy_name, comms = CONFIGS[config]
    ids = [f"r{i}" for i in range(n)]
    env = Environment(
        width=size, height=size, robot_ids=ids, obstacle_density=density,
        max_ticks=max_ticks, enable_comms=comms,
    )
    observations, _ = env.reset(seed=seed)
    policy = make_policy(policy_name)
    rngs = {rid: np.random.default_rng(i) for i, rid in enumerate(ids)}

    def snapshot():
        known = np.full((size, size), -1, dtype=np.int8)
        for robot in env.robots.values():
            known = np.where(known == -1, robot.belief_map, known)
        return known, [env.robots[r].pos for r in ids], env.coverage()

    frames = [snapshot()]
    static = 0
    status = "running"
    while env.agents:
        before = env.successful_moves
        known_before = int((frames[-1][0] != -1).sum())
        actions = policy.act_joint(
            {r: observations[r] for r in env.agents}, {r: rngs[r] for r in env.agents}
        )
        observations, *_ = env.step(actions)
        frames.append(snapshot())
        moved = env.successful_moves != before
        learned = int((frames[-1][0] != -1).sum()) != known_before
        static = 0 if (moved or learned) else static + 1
        if env.agents and static >= DEADLOCK_WINDOW:
            status = "deadlocked"
            break
    if env.coverage_complete():
        status = "complete"
    return env.map.grid.copy(), frames, status


def draw_panel(img, x0, y0, title, grid, frames, t, status, size):
    d = ImageDraw.Draw(img)
    t = min(t, len(frames) - 1)
    known, positions, coverage = frames[t]
    finished = t == len(frames) - 1 and status != "running"

    d.text((x0, y0), title, font=_font(15, bold=True), fill=INK)
    tick_text = f"tick {t}"
    d.text((x0, y0 + 22), tick_text, font=_font(13), fill=INK_2)
    if finished:
        label, color = ("COMPLETE", GOOD) if status == "complete" else ("DEADLOCKED", CRITICAL)
        tw = d.textlength(tick_text, font=_font(13))
        d.text((x0 + tw + 12, y0 + 22), label, font=_font(13, bold=True), fill=color)

    d.text((x0 + size * CELL, y0 + 22), f"{coverage * 100:.0f}% explored", font=_font(13), fill=INK_2, anchor="ra")

    gy = y0 + HEADER - 12
    for r in range(size):
        for c in range(size):
            v = known[r, c]
            color = FOG if v == -1 else (WALL if v == 1 else FREE)
            d.rectangle(
                [x0 + c * CELL, gy + r * CELL, x0 + (c + 1) * CELL - 1, gy + (r + 1) * CELL - 1],
                fill=color,
            )

    # Trails, then robots on top with a surface ring.
    for i in range(len(positions)):
        trail = [
            (x0 + p[1] * CELL + CELL // 2, gy + p[0] * CELL + CELL // 2)
            for p in (f[1][i] for f in frames[: t + 1])
        ]
        if len(trail) > 1:
            d.line(trail, fill=ROBOT_COLORS[i], width=2, joint="curve")
    for i, (r, c) in enumerate(positions):
        cx, cy = x0 + c * CELL + CELL // 2, gy + r * CELL + CELL // 2
        rad = CELL // 2 - 1
        d.ellipse([cx - rad - 2, cy - rad - 2, cx + rad + 2, cy + rad + 2], fill=SURFACE)
        d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=ROBOT_COLORS[i])

    # Coverage meter.
    by = gy + size * CELL + 10
    width = size * CELL
    d.rounded_rectangle([x0, by, x0 + width, by + 8], radius=4, fill=TRACK)
    if coverage > 0:
        d.rounded_rectangle([x0, by, x0 + max(8, int(width * coverage)), by + 8], radius=4, fill=ROBOT_COLORS[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--robots", type=int, default=4)
    parser.add_argument("--size", type=int, default=25)
    parser.add_argument("--density", type=float, default=0.3)
    parser.add_argument("--max-frames", type=int, default=160)
    parser.add_argument("--out", default=os.path.join(REPO, "assets", "demo.gif"))
    args = parser.parse_args()

    panels = [
        ("Independent, no comms", "independent"),
        ("Coordinated + shared maps", "coordinated_comms"),
    ]
    runs = [simulate(cfg, args.seed, args.robots, args.size, args.density, 1500) for _, cfg in panels]
    for (title, _), (_, frames, status) in zip(panels, runs):
        print(f"{title}: {status} after {len(frames) - 1} ticks, coverage {frames[-1][2]:.1%}")

    horizon = max(len(frames) for _, frames, _ in runs)
    stride = max(1, -(-horizon // args.max_frames))
    ticks = list(range(0, horizon, stride)) + [horizon - 1]

    grid_px = args.size * CELL
    w = PAD + len(panels) * (grid_px + PAD * 2) - PAD
    h = PAD + HEADER + grid_px + FOOTER + PAD
    images = []
    for t in ticks:
        img = Image.new("RGB", (w, h), SURFACE)
        for k, ((title, _), (grid, frames, status)) in enumerate(zip(panels, runs)):
            draw_panel(img, PAD + k * (grid_px + PAD * 2), PAD, title, grid, frames, t, status, args.size)
        images.append(img.quantize(colors=64, method=Image.Quantize.MEDIANCUT))

    durations = [70] * (len(images) - 1) + [2500]
    images[0].save(args.out, save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=True)
    print(f"wrote {args.out} ({len(images)} frames, {os.path.getsize(args.out) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
