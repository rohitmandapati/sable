import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from map import Map
from robot import Robot, KNOWN_FREE
from environment import Environment


# Lightweight runner for running multiple iterations of the exploration simulation without rendering
# Outputs average ticks and ticks per free cell for each policy and density combination, good for bench marking


class Runner:
    
    def __init__(
        self,
        move_funcs: list[str],
        size: tuple[int, int],
        seeds: list[int] | None = None,
        densities: float | list[float] = 0.2,
        max_ticks: int = 100_000,
    ):
        self.move_funcs = move_funcs
        self.width, self.height = size
        self.seeds = seeds  # fixed seeds, or None for a random seed each iteration
        self.densities = [densities] if isinstance(densities, (int, float)) else list(densities)
        self.max_ticks = max_ticks

    def _iter_seeds(self, iters: int):
        if self.seeds:
            for seed in self.seeds:
                for _ in range(iters):
                    yield seed
        else:
            for _ in range(iters):
                yield random.randint(0, 2**31 - 1)

    def _run_once(self, move_func: str, seed: int, density: float) -> tuple[int, int] | None:
        # Returns (ticks, free_cells) for a completed run, or None on timeout.
        world = Map(width=self.width, height=self.height, seed=seed, obstacle_density=density)
        free_cells = len(world.free_cells)
        start = tuple(world.free_cells[world.rng.choice(free_cells)])
        robot = Robot(robot_id="runner", pos=start, map_shape=(world.height, world.width))
        env = Environment(map=world, robots=[robot])

        env.observe(robot)  # sense at spawn so the first move respects walls
        explored = lambda: all(robot.belief_map[r, c] == KNOWN_FREE for r, c in world.free_cells)

        while not explored():
            if env.tick_count >= self.max_ticks:
                return None
            env.update_robot_positions(policy=move_func)
            env.tick_count += 1

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


if __name__ == "__main__":

    # Random maps each iteration
    print("--- random seeds ---")
    runner_random = Runner(
        move_funcs=["move_toward_unknown_bfs", "move_random"],
        size=(25, 25),
        seeds=None,
        densities=0.5,
    )
    runner_random.run(iters=20)
