# Benchmark the classical exploration baselines and record per-episode traces.
#
# Two experiments, both fully determined by their seeds:
#
#   1. scaling  -- random 25x25 maps (obstacle density 0.3), team sizes 1/2/4/8,
#                  one root seed per map. Every configuration sees the SAME maps
#                  and the SAME spawns for a given (seed, team size), so
#                  comparisons between configurations are paired.
#   2. suite    -- the 18 handcrafted 10x10 stress-test maps (default_maps.py),
#                  3 robots, many spawn seeds per map.
#
# Episodes mirror runner.Runner: same Environment, same make_policy, same
# per-robot policy-RNG derivation. Two additions the figures need:
#   * the team's coverage is recorded every tick (coverage-over-time curves), and
#   * an episode stops early once it is provably DEADLOCKED: no robot has moved
#     and no robot's belief has changed for DEADLOCK_WINDOW consecutive ticks.
#     The configurations benchmarked here are deterministic functions of
#     (belief, positions), so an unchanged state repeats the same actions forever
#     -- running on to max_ticks would only burn CPU. Deadlocked episodes are
#     recorded as not completed, with the coverage they had reached.
#
# Usage:  python benchmarks/run_benchmarks.py [--seeds 50] [--workers N]
# Output: results/benchmarks/{scaling,suite}.jsonl (raw, one episode per line)

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np

from default_maps import DEFAULT_MAPS
from environment import Environment
from policy import make_policy

# key -> (policy name, comms enabled). The 2x2 of {independent, coordinated} x
# {no comms, shared maps} isolates the value of coordination from the value of
# communication; greedy assignment is the cheaper coordination rule.
CONFIGS: dict[str, tuple[str, bool]] = {
    "independent": ("move_toward_frontier_astar", False),
    "independent_comms": ("move_toward_frontier_astar", True),
    "coordinated": ("coordinated_frontier_hungarian", False),
    "coordinated_comms": ("coordinated_frontier_hungarian", True),
    "greedy_comms": ("coordinated_frontier_greedy", True),
}

DEADLOCK_WINDOW = 10


def run_episode(
    config: str,
    n: int,
    root: int,
    *,
    size: int = 25,
    density: float = 0.3,
    map_name: str | None = None,
    max_ticks: int = 1500,
    policy_seed: int = 0,
) -> dict:
    policy_name, comms = CONFIGS[config]
    robot_ids = [f"r{i}" for i in range(n)]
    env = Environment(
        width=size,
        height=size,
        robot_ids=robot_ids,
        obstacle_density=density,
        max_ticks=max_ticks,
        map_name=map_name,
        enable_comms=comms,
    )
    # Same policy-stream override as Runner._run_episode.
    policy_override = int(
        np.random.SeedSequence([policy_seed, root]).generate_state(1, dtype=np.uint32)[0]
    )
    observations, _ = env.reset(seed=root, options={"seeds": {"policy": policy_override}})
    policy = make_policy(policy_name)
    rngs = {
        rid: np.random.default_rng(child)
        for rid, child in zip(robot_ids, np.random.SeedSequence(env.seeds.policy).spawn(n))
    }

    coverage = [env.coverage()]
    static_ticks = 0
    deadlocked = False
    known_before = sum(int((r.belief_map != -1).sum()) for r in env.robots.values())
    started = time.perf_counter()
    while env.agents:
        moves_before = env.successful_moves
        actions = policy.act_joint(
            {rid: observations[rid] for rid in env.agents},
            {rid: rngs[rid] for rid in env.agents},
        )
        observations, *_ = env.step(actions)
        coverage.append(env.coverage())

        known = sum(int((r.belief_map != -1).sum()) for r in env.robots.values())
        if env.successful_moves == moves_before and known == known_before:
            static_ticks += 1
        else:
            static_ticks = 0
        known_before = known
        if env.agents and static_ticks >= DEADLOCK_WINDOW:
            deadlocked = True
            break

    def first_tick(threshold: float) -> int | None:
        return next((t for t, c in enumerate(coverage) if c >= threshold), None)

    stats = env.comms.stats if env.comms is not None else None
    return {
        "config": config,
        "policy": policy_name,
        "comms": comms,
        "n": n,
        "root": root,
        "map_name": map_name,
        "size": size,
        "density": density,
        "free_cells": int(len(env.map.free_cells)),
        "realized_density": env.map.realized_obstacle_density,
        "completed": env.coverage_complete(),
        "deadlocked": deadlocked,
        "ticks": env.tick_count,
        "max_ticks": max_ticks,
        "final_coverage": coverage[-1],
        "ticks_to_50": first_tick(0.5),
        "ticks_to_90": first_tick(0.9),
        "ticks_to_100": first_tick(1.0),
        "redundancy": env.sensing_redundancy(),
        "successful_moves": env.successful_moves,
        "conflicts": env.conflicts,
        "messages_sent": stats.messages_transmitted if stats else 0,
        "deliveries": stats.deliveries_made if stats else 0,
        "payload_bytes_delivered": stats.payload_bytes_delivered if stats else 0,
        "wire_bytes_delivered": stats.wire_bytes_delivered if stats else 0,
        "wall_seconds": time.perf_counter() - started,
        "coverage": [round(c, 4) for c in coverage],
    }


def _scaling_jobs(seeds: int) -> list[tuple]:
    return [
        (config, n, root, {})
        for config in CONFIGS
        for n in (1, 2, 4, 8)
        for root in range(seeds)
    ]


def _suite_jobs(spawns: int) -> list[tuple]:
    return [
        (config, 3, root, {"size": 10, "map_name": name, "max_ticks": 400})
        for config in CONFIGS
        for name in DEFAULT_MAPS
        for root in range(spawns)
    ]


def _run(job: tuple) -> dict:
    config, n, root, kwargs = job
    return run_episode(config, n, root, **kwargs)


def _execute(jobs: list[tuple], out_path: str, workers: int) -> None:
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool, open(out_path, "w") as f:
        # Longest jobs (big teams, coordinated planners) first for better packing.
        ordered = sorted(jobs, key=lambda j: (-j[1], j[0]))
        for i, record in enumerate(pool.map(_run, ordered, chunksize=1), 1):
            f.write(json.dumps(record) + "\n")
            if i % 100 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} episodes ({time.perf_counter() - started:.0f}s)")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=50, help="random maps per team size")
    parser.add_argument("--spawns", type=int, default=30, help="spawn seeds per handcrafted map")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--only", choices=("scaling", "suite"), default=None)
    args = parser.parse_args()

    out_dir = os.path.join(REPO, "results", "benchmarks")
    os.makedirs(out_dir, exist_ok=True)
    if args.only in (None, "scaling"):
        print(f"scaling: {len(CONFIGS)} configs x 4 team sizes x {args.seeds} maps")
        _execute(_scaling_jobs(args.seeds), os.path.join(out_dir, "scaling.jsonl"), args.workers)
    if args.only in (None, "suite"):
        print(f"suite: {len(CONFIGS)} configs x {len(DEFAULT_MAPS)} maps x {args.spawns} spawns")
        _execute(_suite_jobs(args.spawns), os.path.join(out_dir, "suite.jsonl"), args.workers)
