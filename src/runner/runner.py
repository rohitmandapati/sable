import csv
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, fields

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from environment import Environment
from policy import make_policy

# Coverage thresholds whose time-to-reach every episode records. 100% doubles as
# "time to complete" for episodes that finish.
_COVERAGE_THRESHOLDS: tuple[tuple[str, float], ...] = (
    ("ticks_to_50", 0.50),
    ("ticks_to_90", 0.90),
    ("ticks_to_95", 0.95),
    ("ticks_to_100", 1.00),
)


@dataclass
class EpisodeRecord:

    policy: str
    num_robots: int
    width: int
    height: int

    # Full seed manifest (root + every derived stream) -> episode is replayable.
    root_seed: int
    map_seed: int
    spawn_seed: int
    dynamics_seed: int
    comms_seed: int
    policy_seed: int
    runner_policy_seed: int

    # Map
    map_name: str | None
    requested_density: float
    realized_density: float
    free_cells: int

    # Outcome (timeouts included)
    completed: bool
    final_coverage: float
    ticks: int
    max_ticks: int
    ticks_to_50: int | None
    ticks_to_90: int | None
    ticks_to_95: int | None
    ticks_to_100: int | None

    # Movement / friction
    successful_moves: int
    wall_blocked: int
    conflicts: int
    redundancy: float

    # Communication
    comms_enabled: bool
    comms_drop_prob: float
    messages_attempted: int
    messages_delivered: int
    messages_dropped: int
    bytes_attempted: int
    bytes_delivered: int
    bytes_dropped: int


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
        enable_comms: bool = False,
        comms_drop_prob: float = 0.0,
        comms_max_bytes_per_tick: int | None = None,
    ):
        self.move_funcs = move_funcs
        self.width, self.height = size
        self.seeds = seeds  # explicit deterministic seeds, or None for random
        self.densities = [densities] if isinstance(densities, (int, float)) else list(densities)
        # Robot-count sweep: study how uncoordinated (no-comm) teams scale.
        self.num_robots = [num_robots] if isinstance(num_robots, int) else list(num_robots)
        self.max_ticks = max_ticks
        self.seed_stream = seed_stream
        # Deterministic seed selection: replaces the old global `random` module.
        self._seed_rng = np.random.default_rng(seed_stream)
        self.policy_seed = policy_seed
        # Comms toggle so a sweep can compare coordinated (belief-sharing) teams
        # against the uncoordinated baseline on identical maps.
        self.enable_comms = enable_comms
        self.comms_drop_prob = comms_drop_prob
        self.comms_max_bytes_per_tick = comms_max_bytes_per_tick

        # Populated by run().
        self.records: list[EpisodeRecord] = []
        self.trial_seeds: list[int] = []
        self.dropped_duplicate_seeds = 0

    def _trial_seeds(self, iters: int) -> list[int]:
        # A trial is one root seed. Each root fully determines its episode's
        # derived streams, so the *same root reused for the same config is a
        # literal duplicate* -- we de-duplicate rather than silently inflate the
        # sample count. `iters` only draws that many DISTINCT random roots and is
        # ignored for explicit seeds (repeating them would just duplicate).
        if self.seeds is not None:
            seen: set[int] = set()
            unique: list[int] = []
            for s in self.seeds:
                s = int(s)
                if s not in seen:
                    seen.add(s)
                    unique.append(s)
            self.dropped_duplicate_seeds = len(self.seeds) - len(unique)
            return unique

        self.dropped_duplicate_seeds = 0
        seen = set()
        drawn: list[int] = []
        while len(drawn) < iters:
            s = int(self._seed_rng.integers(0, 2**31 - 1))
            if s in seen:
                continue
            seen.add(s)
            drawn.append(s)
        return drawn

    def _run_episode(
        self, move_func: str, root: int, density: float, n: int,
        map_name: str | None = None,
    ) -> EpisodeRecord:
        # Run a single episode to completion or timeout, always returning a
        # record. Every attempt is preserved. When map_name is given a
        # handcrafted map is used (its own fixed layout/density); density is
        # then only the procedural fallback and is ignored by generation.
        robot_ids = [f"r{i}" for i in range(n)]
        env = Environment(
            width=self.width,
            height=self.height,
            robot_ids=robot_ids,
            obstacle_density=density,
            max_ticks=self.max_ticks,
            map_name=map_name,
            enable_comms=self.enable_comms,
            comms_drop_prob=self.comms_drop_prob,
            comms_max_bytes_per_tick=self.comms_max_bytes_per_tick,
        )

        # Fold the runner's policy_seed into the episode's policy stream via an
        # override, so the manifest's policy seed is the single reproducible
        # source of policy randomness (map/spawn/dynamics/comms stay untouched).
        policy_override = int(
            np.random.SeedSequence([int(self.policy_seed), int(root)]).generate_state(
                1, dtype=np.uint32
            )[0]
        )
        observations, _ = env.reset(seed=root, options={"seeds": {"policy": policy_override}})
        manifest = env.seeds

        policy = make_policy(move_func)
        policy_rngs = {
            rid: np.random.default_rng(child)
            for rid, child in zip(robot_ids, np.random.SeedSequence(manifest.policy).spawn(n))
        }

        # Time-to-coverage: record the first tick each threshold is reached.
        reached: dict[str, int | None] = {name: None for name, _ in _COVERAGE_THRESHOLDS}

        def note_coverage(tick: int) -> None:
            cov = env.coverage()
            for name, thr in _COVERAGE_THRESHOLDS:
                if reached[name] is None and cov >= thr:
                    reached[name] = tick

        note_coverage(0)
        while env.agents:
            # Drive every policy through act_joint: independent policies map it
            # over robots; coordinated policies use the whole team to assign
            # distinct frontiers.
            actions = policy.act_joint(
                {rid: observations[rid] for rid in env.agents},
                {rid: policy_rngs[rid] for rid in env.agents},
            )
            observations, _r, _term, _trunc, _info = env.step(actions)
            note_coverage(env.tick_count)

        completed = env.coverage_complete()
        ch = env.comms
        comms_on = ch is not None
        msgs_delivered = ch.deliveries_made if comms_on else 0
        msgs_dropped = ch.deliveries_dropped if comms_on else 0
        bytes_delivered = ch.payload_bytes_delivered if comms_on else 0
        bytes_dropped = ch.payload_bytes_dropped if comms_on else 0

        return EpisodeRecord(
            policy=move_func,
            num_robots=n,
            width=self.width,
            height=self.height,
            root_seed=manifest.root,
            map_seed=manifest.map,
            spawn_seed=manifest.spawn,
            dynamics_seed=manifest.dynamics,
            comms_seed=manifest.comms,
            policy_seed=manifest.policy,
            runner_policy_seed=self.policy_seed,
            map_name=env.active_map_name,
            requested_density=env.map.requested_obstacle_density,
            realized_density=env.map.realized_obstacle_density,
            free_cells=int(len(env.map.free_cells)),
            completed=completed,
            final_coverage=env.coverage(),
            ticks=env.tick_count,
            max_ticks=self.max_ticks,
            ticks_to_50=reached["ticks_to_50"],
            ticks_to_90=reached["ticks_to_90"],
            ticks_to_95=reached["ticks_to_95"],
            ticks_to_100=reached["ticks_to_100"],
            successful_moves=env.successful_moves,
            wall_blocked=env.wall_bumps,
            conflicts=env.conflicts,
            redundancy=env.sensing_redundancy(),
            comms_enabled=comms_on,
            comms_drop_prob=self.comms_drop_prob,
            messages_attempted=msgs_delivered + msgs_dropped,
            messages_delivered=msgs_delivered,
            messages_dropped=msgs_dropped,
            bytes_attempted=bytes_delivered + bytes_dropped,
            bytes_delivered=bytes_delivered,
            bytes_dropped=bytes_dropped,
        )

    def run(
        self,
        iters: int = 1,
    ) -> dict[str, dict[float, dict[int, dict[str, float]]]]:
        # Run every (policy, density, robot-count, seed) episode, keeping a
        # per-run record for each, then aggregate from those records.
        self.trial_seeds = self._trial_seeds(iters)
        self.records = []
        for move_func in self.move_funcs:
            for density in self.densities:
                for n in self.num_robots:
                    for root in self.trial_seeds:
                        record = self._run_episode(move_func, root, density, n)
                        self.records.append(record)

        results = aggregate_records(self.records)
        _print_summary(results)
        return results

    def config(self, iters: int) -> dict[str, object]:
        # Full configuration + seed provenance, for the machine-readable dump.
        return {
            "move_funcs": list(self.move_funcs),
            "width": self.width,
            "height": self.height,
            "explicit_seeds": None if self.seeds is None else list(self.seeds),
            "trial_seeds": list(self.trial_seeds),
            "dropped_duplicate_seeds": self.dropped_duplicate_seeds,
            "iters": iters,
            "densities": list(self.densities),
            "num_robots": list(self.num_robots),
            "max_ticks": self.max_ticks,
            "seed_stream": self.seed_stream,
            "policy_seed": self.policy_seed,
            "enable_comms": self.enable_comms,
            "comms_drop_prob": self.comms_drop_prob,
            "comms_max_bytes_per_tick": self.comms_max_bytes_per_tick,
        }


# -- aggregation from records ---------------------------------------------


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def aggregate_records(
    records: list[EpisodeRecord],
) -> dict[str, dict[float, dict[int, dict[str, float]]]]:
    """Aggregate per-run records into {policy: {density: {n: metrics}}}.

    Averages over ALL episodes (never dropping timeouts) except `avg_ticks`,
    which is an explicit completed-only completion time paired with
    `completion_rate` and the timeout-inclusive `mean_ticks_capped` so failures
    are always visible.
    """
    results: dict[str, dict[float, dict[int, dict[str, float]]]] = {}
    groups: dict[tuple[str, float, int], list[EpisodeRecord]] = {}
    for r in records:
        groups.setdefault((r.policy, r.requested_density, r.num_robots), []).append(r)

    for (policy, density, n), group in groups.items():
        done = [r for r in group if r.completed]
        runs = len(group)
        avg_ticks = _mean([r.ticks for r in done]) if done else float("inf")
        avg_free = _mean([r.free_cells for r in group])
        ticks_per_cell = (
            avg_ticks / avg_free if done and avg_free else float("inf")
        )

        def reached_mean(attr: str) -> float:
            vals = [getattr(r, attr) for r in group if getattr(r, attr) is not None]
            return _mean(vals) if vals else float("nan")

        metrics: dict[str, float] = {
            # completion-time (completed only) + how many completed, side by side
            "avg_ticks": avg_ticks,
            "completed": len(done),
            "runs": runs,
            "completion_rate": len(done) / runs if runs else float("nan"),
            # timeout-inclusive completion time (timeouts counted at the cap)
            "mean_ticks_capped": _mean(
                [r.ticks if r.completed else r.max_ticks for r in group]
            ),
            "mean_final_coverage": _mean([r.final_coverage for r in group]),
            "avg_free_cells": avg_free,
            "ticks_per_cell": ticks_per_cell,
            "requested_density": density,
            "realized_density": _mean([r.realized_density for r in group]),
            # friction / effort, over all episodes
            "redundancy": _mean([r.redundancy for r in group]),
            "successful_moves": _mean([r.successful_moves for r in group]),
            "wall_bumps": _mean([r.wall_blocked for r in group]),
            "conflicts": _mean([r.conflicts for r in group]),
            "mean_ticks_to_50": reached_mean("ticks_to_50"),
            "mean_ticks_to_90": reached_mean("ticks_to_90"),
            "mean_ticks_to_95": reached_mean("ticks_to_95"),
            # comms
            "comms_enabled": float(any(r.comms_enabled for r in group)),
            "messages_attempted": _mean([r.messages_attempted for r in group]),
            "messages_delivered": _mean([r.messages_delivered for r in group]),
            "messages_dropped": _mean([r.messages_dropped for r in group]),
            "bytes_attempted": _mean([r.bytes_attempted for r in group]),
            "bytes_delivered": _mean([r.bytes_delivered for r in group]),
            "bytes_dropped": _mean([r.bytes_dropped for r in group]),
        }
        results.setdefault(policy, {}).setdefault(density, {})[n] = metrics

    # Speedup vs the single-robot completion time of the same policy+density.
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


def _print_summary(results: dict[str, dict[float, dict[int, dict[str, float]]]]) -> None:
    for policy, per_density in results.items():
        for density, per_n in per_density.items():
            for n, m in per_n.items():
                print(
                    f"{policy} @ density {density} "
                    f"(realized {m['realized_density']:.3f}), n={n}: "
                    f"completion {m['completed']}/{m['runs']} "
                    f"({m['completion_rate'] * 100:.0f}%), "
                    f"avg {m['avg_ticks']:.1f} ticks (completed), "
                    f"capped {m['mean_ticks_capped']:.1f}, "
                    f"final_coverage {m['mean_final_coverage']:.3f}, "
                    f"redundancy {m['redundancy']:.3f}, "
                    f"wall_bumps {m['wall_bumps']:.1f}, conflicts {m['conflicts']:.1f}"
                )


# -- output ----------------------------------------------------------------


def log_results(
    results: dict[str, dict[float, dict[int, dict[str, float]]]],
    size: tuple[int, int],
    results_dir: str | None = None,
    records: list[EpisodeRecord] | None = None,
    config: dict[str, object] | None = None,
) -> str:
    # Human-readable aggregate under results/trial_<n>/... plus, when `records`
    # are supplied, machine-readable per-run output (runs.jsonl + runs.csv) and
    # config.json capturing the full configuration and seed provenance.
    width, height = size
    if results_dir is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        results_dir = os.path.join(repo_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    existing = [d for d in os.listdir(results_dir) if d.startswith("trial_")]
    nums = [int(d.split("_", 1)[1]) for d in existing if d.split("_", 1)[1].isdigit()]
    trial_num = max(nums, default=0) + 1
    trial_dir = os.path.join(results_dir, f"trial_{trial_num}")
    os.makedirs(trial_dir)

    if records is not None:
        _write_records(records, trial_dir)
    if config is not None:
        with open(os.path.join(trial_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2, default=str)

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
                f.write(
                    f"map: {width}x{height}, "
                    f"requested_obstacle_density={density}, "
                    f"realized_obstacle_density={metrics.get('realized_density', float('nan')):.4f}, "
                    f"robots={n}\n"
                )
                f.write(
                    f"completion_rate: {metrics['completion_rate']:.4f} "
                    f"({metrics['completed']}/{metrics['runs']} completed)\n"
                )
                f.write(f"avg_ticks_completed_only: {metrics['avg_ticks']:.2f}\n")
                f.write(f"mean_ticks_capped: {metrics['mean_ticks_capped']:.2f}\n")
                f.write(f"mean_final_coverage: {metrics['mean_final_coverage']:.4f}\n")
                f.write(f"avg_free_cells: {metrics['avg_free_cells']:.2f}\n")
                f.write(f"ticks_per_cell: {metrics['ticks_per_cell']:.4f}\n")
                f.write(f"redundancy: {metrics['redundancy']:.4f}\n")
                f.write(f"successful_moves: {metrics['successful_moves']:.2f}\n")
                f.write(f"wall_bumps: {metrics.get('wall_bumps', 0.0):.2f}\n")
                f.write(f"conflicts: {metrics.get('conflicts', 0.0):.2f}\n")
                f.write(f"speedup_vs_1: {metrics.get('speedup', float('nan')):.4f}\n")
                if metrics.get("comms_enabled"):
                    f.write(
                        f"comms: attempted={metrics['messages_attempted']:.1f} msgs, "
                        f"delivered={metrics['messages_delivered']:.1f}, "
                        f"dropped={metrics['messages_dropped']:.1f}\n"
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
                    f"  {policy} n={n}: completion_rate={mean('completion_rate', rows):.3f}, "
                    f"avg_ticks(completed)={mean('avg_ticks', rows):.2f}, "
                    f"ticks_capped={mean('mean_ticks_capped', rows):.2f}, "
                    f"final_coverage={mean('mean_final_coverage', rows):.4f}, "
                    f"redundancy={mean('redundancy', rows):.4f}, "
                    f"speedup={mean('speedup', rows):.4f}\n"
                )
        f.write("\noverall (all maps, all policies):\n")
        f.write(f"  completion_rate={mean('completion_rate', entries):.4f}\n")
        f.write(f"  avg_ticks_completed_only={mean('avg_ticks', entries):.2f}\n")
        f.write(f"  mean_ticks_capped={mean('mean_ticks_capped', entries):.2f}\n")
        f.write(f"  mean_final_coverage={mean('mean_final_coverage', entries):.4f}\n")
        f.write(f"  avg_free_cells={mean('avg_free_cells', entries):.2f}\n")
        f.write(f"  ticks_per_cell={mean('ticks_per_cell', entries):.4f}\n")
        f.write(f"  redundancy={mean('redundancy', entries):.4f}\n")

    return trial_dir


def _write_records(records: list[EpisodeRecord], trial_dir: str) -> None:
    # One JSON object per line, plus a flat CSV -- both machine-readable and
    # carrying every field (seeds, densities, coverage, comms) of every episode.
    header = [f.name for f in fields(EpisodeRecord)]
    with open(os.path.join(trial_dir, "runs.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")
    with open(os.path.join(trial_dir, "runs.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))


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
        # those episodes are recorded as timeouts (completed=False) rather than
        # spinning to the default.
        max_ticks=5000,
    )
    ITERS = 25
    results = runner_random.run(iters=ITERS)
    trial_dir = log_results(
        results,
        size=(25, 25),
        records=runner_random.records,
        config=runner_random.config(ITERS),
    )
    print(f"logged results to {trial_dir}")

    # Comms on vs off on identical maps (fixed seeds): coordinated belief-sharing
    # should cut redundant exploration versus the uncoordinated baseline.
    print("\n--- comms on vs off (frontier A*, n=4, fixed maps) ---")
    fixed_seeds = list(range(15))
    for label, enable in (("comms OFF", False), ("comms ON (lossless)", True)):
        print(f"[{label}]")
        Runner(
            move_funcs=["move_toward_frontier_astar"],
            size=(25, 25),
            seeds=fixed_seeds,
            densities=0.3,
            num_robots=[4],
            max_ticks=5000,
            enable_comms=enable,
        ).run(iters=1)
