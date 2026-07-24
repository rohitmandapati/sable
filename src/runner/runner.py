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
        max_ticks: int = 100_000,
        seed_stream: int = 0,
        policy_seed: int = 0,
    ):
        self.move_funcs = move_funcs
        self.width, self.height = size
        self.seeds = seeds  # fixed seeds, or None for a random seed each iteration
        self.densities = [densities] if isinstance(densities, (int, float)) else list(densities)
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

    def _run_once(self, move_func: str, seed: int, density: float) -> tuple[int, int] | None:
        # Returns (ticks, free_cells) for a completed run, or None on timeout.
        env = Environment(
            width=self.width,
            height=self.height,
            robot_ids=["runner"],
            obstacle_density=density,
            max_ticks=self.max_ticks,
        )
        policy = make_policy(move_func)
        # Policy RNG is derived from the trial seed so runs are reproducible and
        # independent of which policy is being benchmarked.
        policy_rng = np.random.default_rng(self.policy_seed + seed)

        observations = env.reset(seed=seed)
        free_cells = len(env.map.free_cells)

        terminated = env.coverage_complete()
        truncated = False
        while not (terminated or truncated):
            actions = {
                rid: policy.act(observations[rid], policy_rng)
                for rid in env.active_robot_ids()
            }
            observations, _rewards, terminated, truncated, _info = env.step(actions)

        if truncated:
            return None
        return env.tick_count, free_cells

    def run(
        self,
        iters: int = 1,
    ) -> dict[str, dict[float, dict[str, float]]]:
        results: dict[str, dict[float, dict[str, float]]] = {}

        trial_seeds = list(self._iter_seeds(iters))
        for move_func in self.move_funcs:
            results[move_func] = {}
            for density in self.densities:
                ticks: list[int] = []
                free: list[int] = []
                runs = 0
                # Reuse the same trials for each policy.
                for seed in trial_seeds:
                    runs += 1
                    result = self._run_once(
                        move_func,
                        seed,
                        density,
                    )
                    if result is not None:
                        run_ticks, run_free = result
                        ticks.append(run_ticks)
                        free.append(run_free)
                avg_ticks = (
                    sum(ticks) / len(ticks)
                    if ticks
                    else float("inf")
                )
                avg_free = (
                    sum(free) / len(free)
                    if free
                    else 0.0
                )
                ticks_per_cell = (
                    avg_ticks / avg_free
                    if avg_free
                    else float("inf")
                )
                results[move_func][density] = {
                    "avg_ticks": avg_ticks,
                    "avg_free_cells": avg_free,
                    "ticks_per_cell": ticks_per_cell,
                }
                print(
                    f"{move_func} @ density {density}: "
                    f"avg {avg_ticks:.1f} ticks over "
                    f"{len(ticks)}/{runs} completed runs, "
                    f"avg {avg_free:.1f} free cells, "
                    f"{ticks_per_cell:.2f} ticks/cell"
                )

        return results


def log_results(
    results: dict[str, dict[float, dict[str, float]]],
    size: tuple[int, int],
    results_dir: str | None = None,
) -> str:
    # Write results under results/trial_<n>/map_<size>_d<density>/policy_<name>.log
    # plus an aggregate.log at the trial level averaging everything in the trial.
    width, height = size
    if results_dir is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        results_dir = os.path.join(repo_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Pick the next trial number.
    existing = [d for d in os.listdir(results_dir) if d.startswith("trial_")]
    nums = [int(d.split("_", 1)[1]) for d in existing if d.split("_", 1)[1].isdigit()]
    trial_num = max(nums, default=0) + 1
    trial_dir = os.path.join(results_dir, f"trial_{trial_num}")
    os.makedirs(trial_dir)

    # Reorganize {policy: {density: metrics}} into per-map (density) folders.
    densities = sorted({d for per_density in results.values() for d in per_density})
    entries: list[tuple[float, str, dict[str, float]]] = []
    for density in densities:
        map_dir = os.path.join(trial_dir, f"map_{width}x{height}_d{density}")
        os.makedirs(map_dir)
        for policy, per_density in results.items():
            if density not in per_density:
                continue
            metrics = per_density[density]
            entries.append((density, policy, metrics))
            with open(os.path.join(map_dir, f"policy_{policy}.log"), "w") as f:
                f.write(f"policy: {policy}\n")
                f.write(f"map: {width}x{height}, obstacle_density={density}\n")
                f.write(f"avg_ticks: {metrics['avg_ticks']:.2f}\n")
                f.write(f"avg_free_cells: {metrics['avg_free_cells']:.2f}\n")
                f.write(f"ticks_per_cell: {metrics['ticks_per_cell']:.4f}\n")

    def mean(key: str, rows: list[tuple[float, str, dict[str, float]]]) -> float:
        vals = [m[key] for _, _, m in rows if m[key] != float("inf")]
        return sum(vals) / len(vals) if vals else float("inf")

    with open(os.path.join(trial_dir, "aggregate.log"), "w") as f:
        f.write(f"trial {trial_num} aggregate\n")
        f.write(f"map size: {width}x{height}\n")
        f.write(f"densities: {densities}\n")
        f.write(f"policies: {list(results.keys())}\n\n")
        f.write("per-policy (averaged across maps):\n")
        for policy in results:
            rows = [e for e in entries if e[1] == policy]
            f.write(
                f"  {policy}: avg_ticks={mean('avg_ticks', rows):.2f}, "
                f"avg_free_cells={mean('avg_free_cells', rows):.2f}, "
                f"ticks_per_cell={mean('ticks_per_cell', rows):.4f}\n"
            )
        f.write("\noverall (all maps, all policies):\n")
        f.write(f"  avg_ticks={mean('avg_ticks', entries):.2f}\n")
        f.write(f"  avg_free_cells={mean('avg_free_cells', entries):.2f}\n")
        f.write(f"  ticks_per_cell={mean('ticks_per_cell', entries):.4f}\n")

    return trial_dir


if __name__ == "__main__":

    # Random maps each iteration
    print("--- random seeds ---")
    runner_random = Runner(
        move_funcs=["move_toward_unknown_bfs", "move_toward_frontier_bfs", "move_toward_frontier_astar"],
        size=(25, 25),
        seeds=None,
        densities=0.3,
    )
    results = runner_random.run(iters=100)
    trial_dir = log_results(results, size=(25, 25))
    print(f"logged results to {trial_dir}")
