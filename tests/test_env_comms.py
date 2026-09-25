"""Integration tests for comms wired into the Environment.

conftest.py puts src/ on sys.path, so imports are flat.

Comms is off by default. When on, robots broadcast newly-sensed cells each tick
over the CommsBackend and the receive/trust head fuses whatever is delivered into
belief_map (+ trust_map) ONLY -- never sensed_mask. So comms must:
  - leave physical sensing (sensed_mask, redundancy, coverage-union) untouched
    for an identical action sequence, and
  - enrich per-robot belief with correct teammate cells.

Delivery is NEXT-TICK: a message broadcast at tick t is not delivered until tick
t+1. In particular the initial-sensing deltas sent during reset (tick 0) are not
visible in the reset observations -- they arrive on the first step (age 1).

The backend is the lossless PerfectBroadcastBackend; loss/latency behavior will
be tested against the specific backends that introduce them.
"""

import numpy as np

from actions import Action
from environment import Environment
from robot import UNKNOWN


def _env(enable_comms=False):
    # Open 12x12 map so movement is unobstructed and deterministic per seed.
    return Environment(
        width=12,
        height=12,
        robot_ids=["r0", "r1"],
        obstacle_density=0.0,
        enable_comms=enable_comms,
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
        # ...and nothing was ever fused, so the trust overlay stays empty.
        assert not robot.trust_map.any()


# -- comms on: belief propagates, physical state preserved ----------------------

def test_comms_propagates_correct_teammate_cells_into_belief():
    env = _run_script(_env(enable_comms=True))
    found_received = False
    for robot in env.robots.values():
        received_only = _known(robot) & ~robot.sensed_mask
        if received_only.any():
            found_received = True
            for r, c in np.argwhere(received_only):
                # Received belief must match ground truth (no corruption), must
                # NOT be recorded as first-hand sensing, and must carry trust.
                assert robot.belief_map[r, c] == env.map.grid[r, c]
                assert not robot.sensed_mask[r, c]
                assert robot.trust_map[r, c] == 1.0  # TrustAllReceiver baseline
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


def test_delivery_stats_are_tracked():
    env = _run_script(_env(enable_comms=True))
    stats = env.comms.stats
    assert stats.payload_bytes_delivered > 0
    assert stats.deliveries_made > 0
    # Wire accounting is tracked alongside payload; the frame carries a header, so
    # wire bytes strictly exceed payload bytes for the same traffic.
    assert stats.wire_bytes_transmitted > stats.payload_bytes_transmitted
    assert stats.wire_bytes_delivered > stats.payload_bytes_delivered


# -- tick-zero (initial-sensing) sharing ---------------------------------------

def test_tick_zero_no_comms_belief_equals_sensed():
    # Baseline: with comms off, reset leaves belief == first-hand sensing.
    env = _env(enable_comms=False)
    env.reset(seed=0)  # no steps
    for robot in env.robots.values():
        assert np.array_equal(_known(robot), robot.sensed_mask)


def test_tick_zero_broadcasts_but_delivers_next_tick():
    # Next-tick delivery: the tick-0 deltas are broadcast during reset but NOT
    # delivered yet, so with no steps taken belief still equals first-hand sensing.
    env = _env(enable_comms=True)
    env.reset(seed=0)  # no steps
    for robot in env.robots.values():
        assert np.array_equal(_known(robot), robot.sensed_mask)  # nothing fused yet
    assert env.comms.stats.payload_bytes_transmitted > 0  # ...but the send happened
    assert env.comms.stats.payload_bytes_delivered == 0

    # One STAY step later, the tick-0 broadcasts arrive (age 1) and enrich belief
    # with cells this robot never sensed itself.
    env.step({rid: Action.STAY for rid in env.agents})
    found = False
    for robot in env.robots.values():
        received = _known(robot) & ~robot.sensed_mask
        if received.any():
            found = True
            for r, c in np.argwhere(received):
                assert robot.belief_map[r, c] == env.map.grid[r, c]  # correct
                assert not robot.sensed_mask[r, c]  # never first-hand
    assert found  # two distinct spawns must exchange at least one cell
    assert env.comms.stats.payload_bytes_delivered > 0
