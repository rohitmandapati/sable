"""Integration tests for comms wired into the Environment.

conftest.py puts src/ on sys.path, so imports are flat.

Comms is off by default. When on, robots broadcast newly-sensed cells each tick
and fold whatever the (lossy) channel delivers into belief_map ONLY -- never
sensed_mask. So comms must:
  - leave physical sensing (sensed_mask, redundancy, coverage-union) untouched
    for an identical action sequence, and
  - enrich per-robot belief with correct teammate cells.
"""

import numpy as np

from actions import Action
from environment import Environment
from robot import UNKNOWN


def _env(enable_comms=False, drop_prob=0.0, max_bytes=None):
    # Open 12x12 map so movement is unobstructed and deterministic per seed.
    return Environment(
        width=12,
        height=12,
        robot_ids=["r0", "r1"],
        obstacle_density=0.0,
        enable_comms=enable_comms,
        comms_drop_prob=drop_prob,
        comms_max_bytes_per_tick=max_bytes,
    )


# A fixed wandering script (no policy, so movement is identical regardless of
# comms) that drives the two robots apart, revealing fresh cells to share.
_PATTERNS = {
    "r0": [Action.RIGHT, Action.DOWN, Action.RIGHT, Action.DOWN],
    "r1": [Action.LEFT, Action.UP, Action.LEFT, Action.UP],
}


def _run_script(env, ticks=8, seed=0):
    env.reset(seed=seed)
    for t in range(ticks):
        actions = {rid: pat[t % len(pat)] for rid, pat in _PATTERNS.items()}
        env.step(actions)
    return env


def _known(robot):
    return robot.belief_map != UNKNOWN


# -- comms off (baseline unchanged) --------------------------------------------

def test_comms_off_by_default_belief_equals_sensed():
    env = _run_script(_env(enable_comms=False))
    assert env.comms is None
    for robot in env.robots.values():
        # No sharing -> belief comes only from own sensing.
        assert np.array_equal(_known(robot), robot.sensed_mask)


# -- comms on: belief propagates, physical state preserved ----------------------

def test_comms_propagates_correct_teammate_cells_into_belief():
    env = _run_script(_env(enable_comms=True))
    found_received = False
    for robot in env.robots.values():
        received_only = _known(robot) & ~robot.sensed_mask
        if received_only.any():
            found_received = True
            for r, c in np.argwhere(received_only):
                # Received belief must match ground truth (no corruption) and must
                # NOT be recorded as first-hand sensing.
                assert robot.belief_map[r, c] == env.map.grid[r, c]
                assert not robot.sensed_mask[r, c]
    assert found_received  # lossless comms between two moving robots must share


def test_comms_leaves_physical_sensing_identical_to_baseline():
    # Same seed + same fixed actions => same movement whether or not comms is on
    # (comms touches belief, not positions/_rng). So sensed_mask, redundancy and
    # the coverage union must be bit-for-bit identical.
    off = _run_script(_env(enable_comms=False))
    on = _run_script(_env(enable_comms=True))
    for rid in off.robots:
        assert np.array_equal(off.robots[rid].sensed_mask, on.robots[rid].sensed_mask)
    assert off.sensing_redundancy() == on.sensing_redundancy()
    assert off.coverage() == on.coverage()
    # ...but at least one robot now believes more than it sensed.
    assert any(
        (_known(r) & ~r.sensed_mask).any() for r in on.robots.values()
    )


def test_dropping_everything_matches_no_comms():
    env = _run_script(_env(enable_comms=True, drop_prob=1.0))
    for robot in env.robots.values():
        assert np.array_equal(_known(robot), robot.sensed_mask)  # nothing received
    assert env.comms.bytes_delivered == 0
    assert env.comms.messages_dropped > 0


def test_bandwidth_and_delivery_stats_are_tracked():
    env = _run_script(_env(enable_comms=True))
    assert env.comms.bytes_delivered > 0
