import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from environment import Environment
from policy import make_policy


class Runner:

    def __init__(
        self,
        move_funcs: list[str],
        size: tuple[int, int],
        seeds: list[int] | None = None,
        densities: float | list[float] = 0.2,
        num_robots: int | list[int] = 1,
        max_ticks: int = 100_000,
        seed_stream: int = 0,
        policy_seed: int = 0,
    ):
        self.move_funcs = move_funcs
        self.width, self.height = size
        self.seeds = seeds  # fixed seeds, or None for a random seed each iteration
        self.densities = [densities] if isinstance(densities, (int, float)) else list(densities)
        # Robot-count sweep: study how uncoordinated (no-comm) teams scale.
        self.num_robots = [num_robots] if isinstance(num_robots, int) else list(num_robots)
        self.max_ticks = max_ticks
        # Deterministic seed selection: replaces the old global `random` module.
        self._seed_rng = np.random.default_rng(seed_stream)
        self.policy_seed = policy_seed

    def _iter_seeds(self, iters: int):
        if self.seeds:
            for seed in self.seeds:
                for _ in range(iters):
                    yield seed
        else:
            for _ in range(iters):
                yield int(self._seed_rng.integers(0, 2**31 - 1))

    def _run_once(
        self, move_func: str, seed: int, density: float, n: int
    ) -> tuple[int, int, float] | None:
        # Returns (ticks, free_cells, redundancy) for a completed run, or None on
        # timeout
        robot_ids = [f"r{i}" for i in range(n)]
        env = Environment(
            width=self.width,
            height=self.height,
            robot_ids=robot_ids,
            obstacle_density=density,
            max_ticks=self.max_ticks,
        )
        policy = make_policy(move_func)
        # One shared (stateless) policy, but an independent RNG per robot so that
        # random tie-breaking does not correlate across the team. Entropy is a
        # sequence so distinct (seed, robot) pairs never collide.
        policy_rngs = {
            rid: np.random.default_rng([self.policy_seed, seed, i])
            for i, rid in enumerate(robot_ids)
        }

        observations, _ = env.reset(seed=seed)
        free_cells = len(env.map.free_cells)

        while env.agents:
            actions = {
                rid: policy.act(observations[rid], policy_rngs[rid])
                for rid in env.agents
            }
            observations, _rewards, _terminated, _truncated, _info = env.step(actions)

        truncated = env.tick_count >= self.max_ticks and not env.coverage_complete()
        if truncated:
            return None
        return env.tick_count, free_cells, env.sensing_redundancy()

    def run(
        self,
        iters: int = 1,
    ) -> dict[str, dict[float, dict[int, dict[str, float]]]]:
        results: dict[str, dict[float, dict[int, dict[str, float]]]] = {}

        trial_seeds = list(self._iter_seeds(iters))
        for move_func in self.move_funcs:
            results[move_func] = {}
            for density in self.densities:
                results[move_func][density] = {}
                for n in self.num_robots:
                    ticks: list[int] = []
                    free: list[int] = []
                    redundancy: list[float] = []
                    runs = 0
                    # Reuse the same trials for each policy / robot count.
                    for seed in trial_seeds:
                        runs += 1
                        result = self._run_once(move_func, seed, density, n)
                        if result is not None:
                            run_ticks, run_free, run_red = result
                            ticks.append(run_ticks)
                            free.append(run_free)
                            redundancy.append(run_red)
                    avg_ticks = sum(ticks) / len(ticks) if ticks else float("inf")
                    avg_free = sum(free) / len(free) if free else 0.0
                    ticks_per_cell = avg_ticks / avg_free if avg_free else float("inf")
                    avg_red = sum(redundancy) / len(redundancy) if redundancy else 0.0
                    results[move_func][density][n] = {
                        "avg_ticks": avg_ticks,
                        "avg_free_cells": avg_free,
                        "ticks_per_cell": ticks_per_cell,
                        "redundancy": avg_red,
                        "completed": len(ticks),
                        "runs": runs,
                    }
                    print(
                        f"{move_func} @ density {density}, n={n}: "
                        f"avg {avg_ticks:.1f} ticks over "
                        f"{len(ticks)}/{runs} completed runs, "
                        f"avg {avg_free:.1f} free cells, "
                        f"{ticks_per_cell:.2f} ticks/cell, "
                        f"redundancy {avg_red:.3f}"
                    )

        # Speedup vs the single-robot run of the same policy+density. Computed
        # after the sweep so every robot count is available as a reference.
        for per_density in results.values():
            for per_n in per_density.values():
                base = per_n.get(1, {}).get("avg_ticks")
                for metrics in per_n.values():
                    if (
                        base
                        and base != float("inf")
                        and metrics["avg_ticks"] not in (0, float("inf"))
                    ):
                        metrics["speedup"] = base / metrics["avg_ticks"]
                    else:
                        metrics["speedup"] = float("nan")

        return results


def log_results(
    results: dict[str, dict[float, dict[int, dict[str, float]]]],
    size: tuple[int, int],
    results_dir: str | None = None,
) -> str:
    # Write results under results/trial_<n>/map_<size>_d<density>_n<robots>/policy_<name>.log
    # and an aggregate.log at the trial level averaging everything in the trial
    width, height = size
    if results_dir is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        results_dir = os.path.join(repo_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Pick the next trial number
    existing = [d for d in os.listdir(results_dir) if d.startswith("trial_")]
    nums = [int(d.split("_", 1)[1]) for d in existing if d.split("_", 1)[1].isdigit()]
    trial_num = max(nums, default=0) + 1
    trial_dir = os.path.join(results_dir, f"trial_{trial_num}")
    os.makedirs(trial_dir)

    # Reorganize {policy: {density: {n: metrics}}} into per-map (density, robots) folders
    combos = sorted(
        {
            (density, n)
            for per_density in results.values()
            for density, per_n in per_density.items()
            for n in per_n
        }
    )
    entries: list[tuple[float, int, str, dict[str, float]]] = []
    for density, n in combos:
        map_dir = os.path.join(trial_dir, f"map_{width}x{height}_d{density}_n{n}")
        os.makedirs(map_dir, exist_ok=True)
        for policy, per_density in results.items():
            if density not in per_density or n not in per_density[density]:
                continue
            metrics = per_density[density][n]
            entries.append((density, n, policy, metrics))
            with open(os.path.join(map_dir, f"policy_{policy}.log"), "w") as f:
                f.write(f"policy: {policy}\n")
                f.write(f"map: {width}x{height}, obstacle_density={density}, robots={n}\n")
                f.write(f"avg_ticks: {metrics['avg_ticks']:.2f}\n")
                f.write(f"avg_free_cells: {metrics['avg_free_cells']:.2f}\n")
                f.write(f"ticks_per_cell: {metrics['ticks_per_cell']:.4f}\n")
                f.write(f"redundancy: {metrics['redundancy']:.4f}\n")
                f.write(f"speedup_vs_1: {metrics.get('speedup', float('nan')):.4f}\n")
                f.write(
                    f"completed_runs: {metrics.get('completed', '?')}/{metrics.get('runs', '?')}\n"
                )

    def mean(key: str, rows: list[tuple[float, int, str, dict[str, float]]]) -> float:
        vals = [
            m[key]
            for *_, m in rows
            if key in m
            and m[key] != float("inf")
            and not (isinstance(m[key], float) and math.isnan(m[key]))
        ]
        return sum(vals) / len(vals) if vals else float("inf")

    densities = sorted({d for d, _, _, _ in entries})
    robot_counts = sorted({n for _, n, _, _ in entries})
    with open(os.path.join(trial_dir, "aggregate.log"), "w") as f:
        f.write(f"trial {trial_num} aggregate\n")
        f.write(f"map size: {width}x{height}\n")
        f.write(f"densities: {densities}\n")
        f.write(f"robot_counts: {robot_counts}\n")
        f.write(f"policies: {list(results.keys())}\n\n")
        f.write("per-policy per-robot-count (averaged across densities):\n")
        for policy in results:
            for n in robot_counts:
                rows = [e for e in entries if e[2] == policy and e[1] == n]
                if not rows:
                    continue
                f.write(
                    f"  {policy} n={n}: avg_ticks={mean('avg_ticks', rows):.2f}, "
                    f"ticks_per_cell={mean('ticks_per_cell', rows):.4f}, "
                    f"redundancy={mean('redundancy', rows):.4f}, "
                    f"speedup={mean('speedup', rows):.4f}\n"
                )
        f.write("\noverall (all maps, all policies):\n")
        f.write(f"  avg_ticks={mean('avg_ticks', entries):.2f}\n")
        f.write(f"  avg_free_cells={mean('avg_free_cells', entries):.2f}\n")
        f.write(f"  ticks_per_cell={mean('ticks_per_cell', entries):.4f}\n")
        f.write(f"  redundancy={mean('redundancy', entries):.4f}\n")

    return trial_dir


if __name__ == "__main__":

    # Random maps each iteration, swept across an uncoordinated (no-comm) team.
    print("--- random seeds ---")
    runner_random = Runner(
        move_funcs=["move_toward_unknown_bfs", "move_toward_frontier_bfs", "move_toward_frontier_astar"],
        size=(25, 25),
        seeds=None,
        densities=0.3,
        num_robots=[1, 2, 4, 8],
        # Uncoordinated teams can livelock on the final frontier; cap ticks so
        # those runs are excluded quickly rather than spinning to the default.
        max_ticks=5000,
    )
    results = runner_random.run(iters=25)
    trial_dir = log_results(results, size=(25, 25))
    print(f"logged results to {trial_dir}")
