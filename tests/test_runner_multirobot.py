import math
import os

# src/ is placed on sys.path by conftest.py (same as the app's imports).
from runner.runner import Runner, log_results

POLICY = "move_toward_frontier_bfs"
DENSITY = 0.0  # fully open map -> connected free space -> coverage always completes


def _runner(num_robots):
    return Runner(
        move_funcs=[POLICY],
        size=(10, 10),
        seeds=[1, 2, 3, 4, 5],
        densities=DENSITY,
        num_robots=num_robots,
        # Uncoordinated teams can livelock on the last frontier (two robots race
        # the same cell, collide, re-plan, repeat). A small cap makes those runs
        # bail cheaply; the runner counts them as incomplete and excludes them.
        max_ticks=2000,
    )


def test_multirobot_explores():
    # Teams of 2 and 4 explore a solvable (open) map: at least some runs finish,
    # and completed runs report a finite tick count.
    results = _runner([2, 4]).run(iters=1)
    per = results[POLICY][DENSITY]
    for n in (2, 4):
        metrics = per[n]
        assert metrics["completed"] >= 1
        assert math.isfinite(metrics["avg_ticks"])


def test_single_robot_baseline():
    # A lone robot cannot duplicate work and is its own speedup reference.
    metrics = _runner(1).run(iters=1)[POLICY][DENSITY][1]
    assert metrics["redundancy"] == 0.0
    assert metrics["speedup"] == 1.0


def test_redundancy_grows_with_team():
    per = _runner([1, 4]).run(iters=1)[POLICY][DENSITY]
    assert per[1]["redundancy"] == 0.0
    assert per[4]["redundancy"] > 0.0  # uncoordinated robots re-cover ground


def test_determinism():
    # Identical config -> identical metrics (guards the per-robot RNG wiring).
    a = _runner([2, 4]).run(iters=1)[POLICY][DENSITY]
    b = _runner([2, 4]).run(iters=1)[POLICY][DENSITY]
    for n in (2, 4):
        assert a[n]["avg_ticks"] == b[n]["avg_ticks"]
        assert a[n]["redundancy"] == b[n]["redundancy"]


def test_log_results_writes_sweep(tmp_path):
    results = _runner([1, 2]).run(iters=1)
    trial_dir = log_results(results, size=(10, 10), results_dir=str(tmp_path))

    assert os.path.exists(os.path.join(trial_dir, "aggregate.log"))
    policy_log = os.path.join(
        trial_dir, f"map_10x10_d{DENSITY}_n2", f"policy_{POLICY}.log"
    )
    assert os.path.exists(policy_log)
    with open(policy_log) as f:
        content = f.read()
    assert "redundancy:" in content
    assert "robots=2" in content
