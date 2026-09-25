"""The record-based experiment runner: every attempt is preserved (no survivor
bias), timeouts stay in the results, seed/iters semantics are de-duplicated, and
machine-readable per-run output is written alongside the human aggregate.
"""

import csv
import json
import math
import os
from dataclasses import fields

from runner.runner import EpisodeRecord, Runner, aggregate_records, log_results

POLICY = "move_toward_frontier_bfs"


def _runner(**kwargs):
    params = dict(move_funcs=[POLICY], size=(10, 10), densities=0.0, num_robots=1, max_ticks=2000)
    params.update(kwargs)
    return Runner(**params)


# -- no survivor bias: every attempt recorded, timeouts kept -----------------


def test_every_attempt_is_recorded_even_on_timeout():
    # max_ticks=1 cannot cover a 10x10 room, so every episode times out -- and
    # must still appear as a record with its partial coverage retained.
    r = _runner(seeds=[1, 2, 3], max_ticks=1)
    results = r.run(iters=1)
    assert len(r.records) == 3
    assert all(not rec.completed for rec in r.records)
    for rec in r.records:
        assert 0.0 <= rec.final_coverage < 1.0
        assert rec.ticks == 1

    m = results[POLICY][0.0][1]
    assert m["runs"] == 3
    assert m["completion_rate"] == 0.0
    # Completed-only completion time is inf (none completed) -- failures are not
    # hidden -- while the capped metric and final coverage remain finite.
    assert math.isinf(m["avg_ticks"])
    assert m["mean_ticks_capped"] == 1.0
    assert not math.isnan(m["mean_final_coverage"])


def test_completion_rate_and_capped_metric_are_separate():
    r = _runner(seeds=[1, 2, 3, 4], max_ticks=2000)
    results = r.run(iters=1)
    m = results[POLICY][0.0][1]
    assert m["runs"] == 4
    assert 0.0 <= m["completion_rate"] <= 1.0
    if m["completed"]:
        # Capped mean never undercuts the completed-only mean (timeouts sit at
        # the cap, which is >= any completed tick count).
        assert m["mean_ticks_capped"] >= m["avg_ticks"] or math.isinf(m["avg_ticks"])


# -- seeds / iters semantics -------------------------------------------------


def test_duplicate_explicit_seeds_are_deduped():
    r = _runner(seeds=[1, 1, 2, 2, 2], max_ticks=500)
    r.run(iters=1)
    assert r.dropped_duplicate_seeds == 3
    assert sorted(r.trial_seeds) == [1, 2]
    assert len(r.records) == 2


def test_iters_does_not_inflate_explicit_seeds():
    r = _runner(seeds=[1, 2], max_ticks=500)
    r.run(iters=5)  # would be 10 duplicated episodes under the old semantics
    assert len(r.records) == 2


def test_random_seeds_are_distinct_per_iter():
    r = _runner(seeds=None, max_ticks=500)
    r.run(iters=4)
    assert len({rec.root_seed for rec in r.records}) == 4


# -- record completeness -----------------------------------------------------


def test_record_carries_full_seed_manifest_and_distinct_streams():
    r = _runner(seeds=[7], max_ticks=2000)
    r.run(iters=1)
    rec = r.records[0]
    streams = [rec.map_seed, rec.spawn_seed, rec.dynamics_seed, rec.comms_seed, rec.policy_seed]
    assert len(set(streams)) == len(streams)  # independent streams
    assert rec.root_seed == 7
    assert rec.requested_density == 0.0
    assert rec.realized_density != 0.0  # borders + pruning add walls


def test_coverage_thresholds_are_monotonic_for_completed_run():
    r = _runner(seeds=[3], max_ticks=2000)
    r.run(iters=1)
    rec = r.records[0]
    assert rec.completed
    assert rec.final_coverage == 1.0
    stamps = [rec.ticks_to_50, rec.ticks_to_90, rec.ticks_to_95, rec.ticks_to_100]
    reached = [s for s in stamps if s is not None]
    assert reached == sorted(reached)  # non-decreasing
    assert rec.ticks_to_100 == rec.ticks  # 100% is completion time


def test_comms_stats_recorded_when_enabled():
    r = _runner(num_robots=2, seeds=[1], max_ticks=2000, enable_comms=True)
    r.run(iters=1)
    rec = r.records[0]
    assert rec.comms_enabled
    # Lossless backend: every broadcast reaches the teammate.
    assert rec.messages_delivered > 0  # two robots share sensed cells each tick
    assert rec.bytes_delivered > 0


def test_comms_disabled_leaves_zero_stats():
    r = _runner(num_robots=2, seeds=[1], max_ticks=2000, enable_comms=False)
    r.run(iters=1)
    rec = r.records[0]
    assert rec.comms_enabled is False
    assert rec.messages_delivered == 0 and rec.bytes_delivered == 0


# -- machine-readable output -------------------------------------------------


def test_writes_jsonl_csv_and_config(tmp_path):
    r = _runner(seeds=[1, 2], num_robots=2, max_ticks=2000)
    results = r.run(iters=1)
    trial_dir = log_results(
        results, size=(10, 10), results_dir=str(tmp_path),
        records=r.records, config=r.config(1),
    )

    jsonl = os.path.join(trial_dir, "runs.jsonl")
    csv_path = os.path.join(trial_dir, "runs.csv")
    cfg_path = os.path.join(trial_dir, "config.json")
    assert os.path.exists(os.path.join(trial_dir, "aggregate.log"))
    assert os.path.exists(jsonl) and os.path.exists(csv_path) and os.path.exists(cfg_path)

    record_field_names = {f.name for f in fields(EpisodeRecord)}
    with open(jsonl) as f:
        rows = [json.loads(line) for line in f]
    assert len(rows) == len(r.records)
    assert set(rows[0]) == record_field_names  # every field serialized

    with open(csv_path) as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == len(r.records)

    with open(cfg_path) as f:
        cfg = json.load(f)
    assert cfg["trial_seeds"] == [1, 2]
    assert cfg["dropped_duplicate_seeds"] == 0
    assert cfg["densities"] == [0.0]


def test_aggregate_records_is_pure_and_groups_by_config():
    r = _runner(seeds=[1, 2, 3], num_robots=1, max_ticks=2000)
    r.run(iters=1)
    agg = aggregate_records(r.records)
    assert set(agg) == {POLICY}
    assert agg[POLICY][0.0][1]["runs"] == 3
