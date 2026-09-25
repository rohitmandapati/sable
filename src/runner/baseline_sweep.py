# Note, this file is largely
# vibe coded, simple script to
# parallelize the baseline sweep
# on my Apple M2 silicon chip.

# This is just for baseline results,
# not comms policy training.


"""Major classical baseline sweep for SABLE.

Produces the reference numbers the learned (MAPPO) comms policy will be measured
against. Three deliberately-chosen blocks, all driven through the canonical
`Runner._run_episode` (so seeding/metrics are identical to the normal harness):

  A. Scaling & policy comparison  -- comms OFF, procedural 25x25, n in {1,2,4,8}.
       Establishes classical no-comms baselines + speedup curves.
  B. Comms degradation ladder     -- procedural 25x25, n=4, comms
       {OFF, lossless, drop 0.3, drop 0.7}. The rungs a smart policy must beat.
  C. Structured maps              -- all handcrafted 10x10 maps, n in {2,4},
       comms {OFF, lossless}. Topology stress (corridors, traps, rooms).

Episodes are fully independent and self-seeded, so the sweep is run across a
process pool; results are identical to a serial run (per-episode RNG depends
only on the root seed, never on execution order). Each record is streamed to
runs.jsonl as it completes (crash-safe), then re-dumped as runs.csv with
config.json (matrix + provenance) and summary.md (human-readable tables).
Nothing here is committed -- it writes under results/.

Run:  .venv/bin/python -m src.runner.baseline_sweep
"""

from __future__ import annotations

import os

# Pin the numerical backends to one thread each BEFORE numpy is imported (via
# the runner import below): the parallelism here is across processes/episodes,
# so per-process BLAS threads would only oversubscribe the cores and slow us
# down. On Apple silicon numpy uses Accelerate (VECLIB).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from dataclasses import asdict, fields  # noqa: E402
from multiprocessing import Pool  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from default_maps import DEFAULT_MAPS  # noqa: E402
from runner.runner import EpisodeRecord, Runner  # noqa: E402

# -- matrix ----------------------------------------------------------------

UNCOORD = ["move_toward_frontier_astar", "move_toward_unknown_bfs"]
COORD = ["coordinated_frontier_greedy", "coordinated_frontier_hungarian"]

# Block A: classical scaling (comms off).
A_POLICIES = UNCOORD + COORD
A_SIZE = (25, 25)
A_DENSITY = 0.30
A_ROBOTS = [1, 2, 4, 8]
A_SEEDS = list(range(8))
A_MAX_TICKS = 3000

# Block B: comms degradation ladder. astar is comms-enriched-but-uncoordinated;
# the two coordinated policies are the real belief-sharing baselines.
B_POLICIES = ["move_toward_frontier_astar"] + COORD
B_SIZE = (25, 25)
B_DENSITY = 0.30
B_ROBOTS = 4
B_SEEDS = list(range(10))
B_MAX_TICKS = 3000
# (enable_comms, drop_prob, label)
B_LADDER = [
    (False, 0.0, "off"),
    (True, 0.0, "lossless"),
    (True, 0.3, "drop0.3"),
    (True, 0.7, "drop0.7"),
]

# Block C: structured maps (all 10x10, their own fixed density).
C_POLICIES = UNCOORD + COORD
C_SIZE = (10, 10)
C_MAPS = list(DEFAULT_MAPS)
C_ROBOTS = [2, 4]
C_SEEDS = list(range(5))
C_MAX_TICKS = 1500
C_COMMS = [(False, 0.0, "off"), (True, 0.0, "lossless")]

N_WORKERS = max(1, (os.cpu_count() or 2) - 1)


# -- job model -------------------------------------------------------------
# A job is a flat dict of primitives so it pickles cleanly to pool workers.


def _build_jobs() -> list[dict]:
    jobs: list[dict] = []
    for pol in A_POLICIES:
        for n in A_ROBOTS:
            for s in A_SEEDS:
                jobs.append(dict(tag="A/scale", size=A_SIZE, max_ticks=A_MAX_TICKS,
                                 enable_comms=False, drop=0.0, policy=pol,
                                 seed=s, density=A_DENSITY, n=n, map_name=None))
    for enable, drop, label in B_LADDER:
        for pol in B_POLICIES:
            for s in B_SEEDS:
                jobs.append(dict(tag=f"B/{label}", size=B_SIZE, max_ticks=B_MAX_TICKS,
                                 enable_comms=enable, drop=drop, policy=pol,
                                 seed=s, density=B_DENSITY, n=B_ROBOTS, map_name=None))
    for enable, drop, label in C_COMMS:
        for map_name in C_MAPS:
            for pol in C_POLICIES:
                for n in C_ROBOTS:
                    for s in C_SEEDS:
                        jobs.append(dict(tag=f"C/{label}/{map_name[:10]}", size=C_SIZE,
                                         max_ticks=C_MAX_TICKS, enable_comms=enable,
                                         drop=drop, policy=pol, seed=s, density=0.0,
                                         n=n, map_name=map_name))
    return jobs


def _run_job(job: dict) -> tuple[str, EpisodeRecord]:
    # Reconstruct a Runner for this job's config and run the one episode through
    # the canonical harness. Cheap: Runner construction is trivial.
    r = Runner(
        move_funcs=[], size=job["size"], max_ticks=job["max_ticks"],
        enable_comms=job["enable_comms"], comms_drop_prob=job["drop"],
    )
    rec = r._run_episode(
        job["policy"], job["seed"], job["density"], job["n"],
        map_name=job["map_name"],
    )
    return job["tag"], rec


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir = os.path.join(repo_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    existing = [d for d in os.listdir(results_dir) if d.startswith("baseline_")]
    nums = [int(d.split("_", 1)[1]) for d in existing if d.split("_", 1)[1].isdigit()]
    trial = max(nums, default=0) + 1
    out_dir = os.path.join(results_dir, f"baseline_{trial}")
    os.makedirs(out_dir)

    jobs = _build_jobs()
    total = len(jobs)
    print(f"baseline_{trial}: {total} episodes across {N_WORKERS} workers", flush=True)

    records: list[EpisodeRecord] = []
    jsonl = open(os.path.join(out_dir, "runs.jsonl"), "w")
    done = 0
    # Chunksize 1 so the long (n=8 coordinated) episodes are load-balanced across
    # workers rather than clumping in one worker's chunk.
    with Pool(N_WORKERS) as pool:
        for tag, rec in pool.imap_unordered(_run_job, jobs, chunksize=1):
            records.append(rec)
            jsonl.write(json.dumps(asdict(rec)) + "\n")
            jsonl.flush()
            done += 1
            status = "OK " if rec.completed else "cap"
            print(
                f"[{done:4d}/{total}] {tag:22} {rec.policy[:26]:26} "
                f"n={rec.num_robots} seed={rec.root_seed} {status} "
                f"cov={rec.final_coverage:.3f} ticks={rec.ticks} "
                f"redun={rec.redundancy:.3f}",
                flush=True,
            )
    jsonl.close()

    header = [f.name for f in fields(EpisodeRecord)]
    with open(os.path.join(out_dir, "runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for rec in records:
            w.writerow(asdict(rec))

    config = {
        "git_commit": _git_commit(),
        "n_workers": N_WORKERS,
        "total_episodes": len(records),
        "block_A": {
            "policies": A_POLICIES, "size": A_SIZE, "density": A_DENSITY,
            "robots": A_ROBOTS, "seeds": A_SEEDS, "max_ticks": A_MAX_TICKS,
            "comms": "off",
        },
        "block_B": {
            "policies": B_POLICIES, "size": B_SIZE, "density": B_DENSITY,
            "robots": B_ROBOTS, "seeds": B_SEEDS, "max_ticks": B_MAX_TICKS,
            "ladder": [{"enable_comms": e, "drop": d, "label": l} for e, d, l in B_LADDER],
        },
        "block_C": {
            "policies": C_POLICIES, "size": C_SIZE, "maps": C_MAPS,
            "robots": C_ROBOTS, "seeds": C_SEEDS, "max_ticks": C_MAX_TICKS,
            "comms": [{"enable_comms": e, "drop": d, "label": l} for e, d, l in C_COMMS],
        },
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    _write_summary(records, out_dir)
    print(f"\nbaseline written to {out_dir}", flush=True)


# -- summary ---------------------------------------------------------------


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def _agg(group: list[EpisodeRecord]) -> dict:
    done = [r for r in group if r.completed]
    return {
        "runs": len(group),
        "completed": len(done),
        "completion_rate": len(done) / len(group) if group else float("nan"),
        "avg_ticks_done": _mean([r.ticks for r in done]) if done else float("nan"),
        "ticks_capped": _mean([r.ticks if r.completed else r.max_ticks for r in group]),
        "final_cov": _mean([r.final_coverage for r in group]),
        "redundancy": _mean([r.redundancy for r in group]),
        "conflicts": _mean([r.conflicts for r in group]),
        "bytes_delivered": _mean([r.bytes_delivered for r in group]),
        "bytes_dropped": _mean([r.bytes_dropped for r in group]),
    }


def _table(f, rows: list[tuple[str, dict]], key_header: str) -> None:
    f.write(
        f"| {key_header} | runs | compl. | avg ticks (done) | ticks capped | "
        f"final cov | redundancy | conflicts | bytes deliv/drop |\n"
    )
    f.write("|" + "---|" * 9 + "\n")
    for key, m in rows:
        f.write(
            f"| {key} | {m['runs']} | {m['completed']}/{m['runs']} "
            f"({m['completion_rate']*100:.0f}%) | {m['avg_ticks_done']:.1f} | "
            f"{m['ticks_capped']:.1f} | {m['final_cov']:.3f} | "
            f"{m['redundancy']:.3f} | {m['conflicts']:.2f} | "
            f"{m['bytes_delivered']:.0f}/{m['bytes_dropped']:.0f} |\n"
        )
    f.write("\n")


def _write_summary(records: list[EpisodeRecord], out_dir: str) -> None:
    procedural = [r for r in records if r.map_name is None]
    named = [r for r in records if r.map_name is not None]
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("# SABLE classical baseline\n\n")
        f.write(f"total episodes: {len(records)}\n\n")

        f.write("## Block A -- classical scaling (comms OFF, 25x25, d=0.30)\n\n")
        rows = []
        for pol in A_POLICIES:
            for n in A_ROBOTS:
                g = [r for r in procedural if r.policy == pol and r.num_robots == n
                     and not r.comms_enabled]
                if g:
                    rows.append((f"{pol} n={n}", _agg(g)))
        _table(f, rows, "policy / n")

        f.write("## Block B -- comms ladder (25x25, n=4, d=0.30)\n\n")
        rows = []
        for pol in B_POLICIES:
            for enable, drop, label in B_LADDER:
                g = [r for r in procedural if r.policy == pol and r.num_robots == 4
                     and r.comms_enabled == enable
                     and (not enable or r.comms_drop_prob == drop)]
                if g:
                    rows.append((f"{pol} [{label}]", _agg(g)))
        _table(f, rows, "policy / comms")

        f.write("## Block C -- structured maps (10x10), per map (comms OFF)\n\n")
        rows = []
        for map_name in C_MAPS:
            g = [r for r in named if r.map_name == map_name and not r.comms_enabled]
            if g:
                rows.append((map_name, _agg(g)))
        _table(f, rows, "map (all policies/n, comms off)")

        f.write("## Block C -- comms OFF vs lossless (10x10, aggregated over maps)\n\n")
        rows = []
        for pol in C_POLICIES:
            for n in C_ROBOTS:
                for enable, drop, label in C_COMMS:
                    g = [r for r in named if r.policy == pol and r.num_robots == n
                         and r.comms_enabled == enable]
                    if g:
                        rows.append((f"{pol} n={n} [{label}]", _agg(g)))
        _table(f, rows, "policy / n / comms")


if __name__ == "__main__":
    main()
 