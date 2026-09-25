"""Unit tests for the MAPPO-facing comms observation builder.

conftest.py puts src/ on sys.path, so imports are flat.

These pin the contract the learned comms policy slots into: every output has a
FIXED shape regardless of how many messages landed, the mask correctly marks real
vs padding rows, the zero-message case is valid (not a crash), and the message
ordering is DETERMINISTIC -- independent of the order messages were delivered in.
"""

import numpy as np
import pytest

from comms import CommsStats, PayloadKind
from comms.delivered import DeliveredMessage
from config import (
    FRONTIER_SUMMARY_D,
    MAX_MESSAGES,
    MESSAGE_D,
    RESOURCE_D,
    ROBOT_D,
)
from learning import build_comms_observation
from learning.comms_observation import CommsObservation
from robot import KNOWN_FREE, KNOWN_WALL, UNKNOWN, Robot


def _delivered(cells, *, sender="a", created=1, delivered=None, seq=0, sender_pos=None):
    # delivered defaults to created + 1: next-tick delivery, so age == 1, exactly
    # as the backends produce it.
    if delivered is None:
        delivered = created + 1
    return DeliveredMessage(
        sender_id=sender,
        payload_kind=PayloadKind.BELIEF_DELTA,
        cells=tuple(cells),
        created_tick=created,
        delivered_tick=delivered,
        sequence_id=seq,
        payload_size_bytes=len(cells) * 6,
        sender_position=sender_pos,
    )


def _build(robot, *, messages=(), stats=None, tick=2, **kw):
    return build_comms_observation(
        belief_map=robot.belief_map,
        sensed_mask=robot.sensed_mask,
        trust_map=robot.trust_map,
        position=robot.position,
        delivered_messages=messages,
        comms_stats=stats if stats is not None else CommsStats(),
        tick=tick,
        **kw,
    )


# -- shapes / dtypes -----------------------------------------------------------

def test_output_shapes_and_dtypes_are_fixed():
    robot = Robot(robot_id="r", pos=(0, 0), map_shape=(6, 8))
    obs = _build(robot, messages=[_delivered([((1, 1), KNOWN_FREE)])])
    assert isinstance(obs, CommsObservation)
    assert obs.robot_state.shape == (ROBOT_D,)
    assert obs.message_features.shape == (MAX_MESSAGES, MESSAGE_D)
    assert obs.message_mask.shape == (MAX_MESSAGES,)
    assert obs.resource_features.shape == (RESOURCE_D,)
    assert obs.frontier_summary.shape == (FRONTIER_SUMMARY_D,)
    for arr in (
        obs.robot_state,
        obs.message_features,
        obs.message_mask,
        obs.resource_features,
        obs.frontier_summary,
    ):
        assert arr.dtype == np.float32


def test_shape_is_independent_of_message_count():
    robot = Robot(robot_id="r", pos=(0, 0), map_shape=(6, 6))
    few = _build(robot, messages=[_delivered([((1, 1), KNOWN_FREE)])])
    many = _build(
        robot,
        messages=[_delivered([((0, c), KNOWN_FREE)], seq=c) for c in range(6)],
    )
    assert few.message_features.shape == many.message_features.shape == (MAX_MESSAGES, MESSAGE_D)


def test_outputs_are_read_only():
    robot = Robot(robot_id="r", pos=(0, 0), map_shape=(4, 4))
    obs = _build(robot, messages=[_delivered([((1, 1), KNOWN_FREE)])])
    for arr in (obs.robot_state, obs.message_features, obs.message_mask,
                obs.resource_features, obs.frontier_summary):
        with pytest.raises(ValueError):
            arr[...] = 0


# -- masks ---------------------------------------------------------------------

def test_mask_marks_real_rows_and_pads_the_rest():
    robot = Robot(robot_id="r", pos=(0, 0), map_shape=(6, 6))
    msgs = [_delivered([((0, c), KNOWN_FREE)], seq=c) for c in range(3)]
    obs = _build(robot, messages=msgs)
    assert obs.message_mask.sum() == 3
    assert obs.num_messages == 3
    assert list(obs.message_mask[:3]) == [1.0, 1.0, 1.0]
    assert not obs.message_mask[3:].any()
    # Padding rows are all-zeros; real rows are not (each carries a staleness value).
    assert not obs.message_features[3:].any()
    assert obs.message_features[:3].any()


def test_overflow_keeps_max_messages_and_full_mask():
    robot = Robot(robot_id="r", pos=(0, 0), map_shape=(30, 30))
    msgs = [
        _delivered([((0, 0), KNOWN_FREE)], sender="a", created=t, seq=t)
        for t in range(MAX_MESSAGES + 5)
    ]
    obs = _build(robot, messages=msgs, tick=MAX_MESSAGES + 6)
    assert obs.message_mask.sum() == MAX_MESSAGES
    assert obs.message_mask.all()


def test_overflow_keeps_the_freshest_messages():
    robot = Robot(robot_id="r", pos=(0, 0), map_shape=(10, 10))
    total = 100
    # created ticks 0..MAX_MESSAGES, each with a distinct payload size (t + 1
    # cells) so the row order and membership are readable off feature column 1.
    msgs = [
        _delivered([((0, c), KNOWN_FREE) for c in range(t + 1)], sender="a", created=t, seq=t)
        for t in range(MAX_MESSAGES + 1)
    ]
    obs = _build(robot, messages=msgs, tick=MAX_MESSAGES + 2)
    # The oldest message (created tick 0, size 1) is dropped: row 0 is now the
    # created-tick-1 message (size 2), and no surviving row has the size-1 payload.
    assert obs.message_features[0, 1] == pytest.approx(2 / total)
    assert not np.any(np.isclose(obs.message_features[:, 1], 1 / total))


# -- zero-message case ---------------------------------------------------------

def test_zero_messages_is_valid_all_zero_block():
    robot = Robot(robot_id="r", pos=(0, 0), map_shape=(5, 5))
    obs = _build(robot, messages=[])
    assert obs.message_mask.sum() == 0
    assert not obs.message_features.any()
    assert obs.num_messages == 0
    # The rest of the observation is still well-formed.
    assert obs.robot_state.shape == (ROBOT_D,)
    assert obs.resource_features.shape == (RESOURCE_D,)


def test_empty_payload_message_still_masks_as_real():
    robot = Robot(robot_id="r", pos=(0, 0), map_shape=(5, 5))
    obs = _build(robot, messages=[_delivered([])])  # a send carrying nothing new
    assert obs.message_mask[0] == 1.0
    assert obs.message_mask.sum() == 1
    # Fraction features are 0 with no cells, but staleness is still populated.
    assert obs.message_features[0, 0] > 0.0


# -- deterministic ordering ----------------------------------------------------

def test_ordering_is_independent_of_delivery_order():
    robot = Robot(robot_id="r", pos=(0, 0), map_shape=(6, 6))
    msgs = [
        _delivered([((0, 0), KNOWN_FREE)], sender="b", created=2, seq=0),
        _delivered([((0, 1), KNOWN_FREE)], sender="a", created=1, seq=5),
        _delivered([((0, 2), KNOWN_FREE)], sender="a", created=1, seq=2),
    ]
    forward = _build(robot, messages=msgs, tick=5)
    reversed_ = _build(robot, messages=list(reversed(msgs)), tick=5)
    assert np.array_equal(forward.message_features, reversed_.message_features)
    assert np.array_equal(forward.message_mask, reversed_.message_mask)


def test_ordering_is_created_tick_then_sender_then_seq():
    robot = Robot(robot_id="r", pos=(0, 0), map_shape=(6, 6))
    # Distinct payload sizes let us read the row order off feature column 1.
    msgs = [
        _delivered([((0, 0), KNOWN_FREE)] * 1, sender="b", created=1, seq=9),  # 1 cell
        _delivered([((0, c), KNOWN_FREE) for c in range(3)], sender="a", created=1, seq=1),  # 3 cells
        _delivered([((0, c), KNOWN_FREE) for c in range(2)], sender="a", created=1, seq=4),  # 2 cells
    ]
    obs = _build(robot, messages=msgs, tick=5)
    # Expected order: (a,seq1)->3 cells, (a,seq4)->2 cells, (b,seq9)->1 cell.
    total = 6 * 6
    assert obs.message_features[0, 1] == pytest.approx(3 / total)
    assert obs.message_features[1, 1] == pytest.approx(2 / total)
    assert obs.message_features[2, 1] == pytest.approx(1 / total)


def test_full_build_is_deterministic():
    robot = Robot(robot_id="r", pos=(1, 2), map_shape=(6, 6))
    robot.belief_map[0, 0] = KNOWN_FREE
    robot.sensed_mask[0, 0] = True
    msgs = [_delivered([((3, 3), KNOWN_WALL)], sender="a", created=1, seq=0)]
    a = _build(robot, messages=msgs)
    b = _build(robot, messages=msgs)
    for name in ("robot_state", "message_features", "message_mask",
                 "resource_features", "frontier_summary"):
        assert np.array_equal(getattr(a, name), getattr(b, name))


# -- message feature semantics -------------------------------------------------

def test_novel_agree_conflict_fractions_against_belief():
    robot = Robot(robot_id="r", pos=(0, 0), map_shape=(6, 6))
    robot.belief_map[0, 1] = KNOWN_FREE   # message will AGREE here
    robot.belief_map[0, 2] = KNOWN_FREE   # message will CONFLICT here (claims WALL)
    msg = _delivered([
        ((0, 0), KNOWN_FREE),   # novel (belief UNKNOWN)
        ((0, 1), KNOWN_FREE),   # agree
        ((0, 2), KNOWN_WALL),   # conflict
    ])
    obs = _build(robot, messages=[msg])
    novel, agree, conflict = obs.message_features[0, 3:6]
    assert novel == pytest.approx(1 / 3)
    assert agree == pytest.approx(1 / 3)
    assert conflict == pytest.approx(1 / 3)


def test_sender_position_features():
    robot = Robot(robot_id="r", pos=(0, 0), map_shape=(10, 10))
    with_pos = _build(robot, messages=[_delivered([((0, 0), KNOWN_FREE)], sender_pos=(2, 3))])
    without = _build(robot, messages=[_delivered([((0, 0), KNOWN_FREE)], sender_pos=None)])
    # column 7 is the has-position flag; column 6 is the normalized distance.
    assert with_pos.message_features[0, 7] == 1.0
    assert without.message_features[0, 7] == 0.0
    assert with_pos.message_features[0, 6] == pytest.approx((2 + 3) / (10 + 10))
    assert without.message_features[0, 6] == 0.0


def test_message_features_stay_bounded():
    robot = Robot(robot_id="r", pos=(0, 0), map_shape=(4, 4))
    # A payload far larger than the map + out-of-bounds sender must stay in [0, 1].
    huge = [((r, c), KNOWN_FREE) for r in range(4) for c in range(4)] * 5
    obs = _build(robot, messages=[_delivered(huge, sender_pos=(99, 99))])
    row = obs.message_features[0]
    assert np.all(row >= 0.0) and np.all(row <= 1.0)


# -- robot_state / resource / frontier -----------------------------------------

def test_robot_state_reflects_position_and_coverage():
    robot = Robot(robot_id="r", pos=(5, 9), map_shape=(6, 10))
    robot.belief_map[0, 0] = KNOWN_FREE
    robot.belief_map[0, 1] = KNOWN_WALL
    robot.sensed_mask[0, 0] = True
    obs = _build(robot)
    assert obs.robot_state[0] == pytest.approx(5 / 5)   # row / (H-1)
    assert obs.robot_state[1] == pytest.approx(9 / 9)   # col / (W-1)
    assert obs.robot_state[2] == pytest.approx(2 / 60)  # known fraction
    assert obs.robot_state[3] == pytest.approx(1 / 60)  # sensed fraction


def test_resource_features_track_stats_and_trust():
    robot = Robot(robot_id="r", pos=(0, 0), map_shape=(4, 4))
    robot.fuse_cell((0, 0), KNOWN_FREE, trust=1.0)  # one cell trusted -> mean 1/16
    stats = CommsStats(messages_transmitted=4, payload_bytes_transmitted=1000)
    obs = _build(robot, stats=stats, tick=8, byte_scale=2000.0, tick_scale=16.0)
    assert obs.resource_features[0] == pytest.approx(8 / 16)     # elapsed-time budget
    assert obs.resource_features[1] == pytest.approx(4 / 8)      # send rate / tick
    assert obs.resource_features[2] == pytest.approx(1000 / 2000)  # payload bytes spent
    assert obs.resource_features[3] == pytest.approx(1 / 16)     # mean comms trust


def test_frontier_summary_toggle():
    robot = Robot(robot_id="r", pos=(0, 0), map_shape=(6, 6))
    # A single known-free cell adjacent to unknowns is a frontier.
    robot.belief_map[0, 0] = KNOWN_FREE
    on = _build(robot, include_frontiers=True)
    off = _build(robot, include_frontiers=False)
    assert on.frontier_summary[5] == 1.0    # any-frontier flag set
    assert not off.frontier_summary.any()   # disabled -> all zeros


def test_frontier_summary_zero_when_no_frontiers():
    robot = Robot(robot_id="r", pos=(0, 0), map_shape=(4, 4))
    # Fully known belief -> no frontier cells.
    robot.belief_map[:] = KNOWN_FREE
    obs = _build(robot, include_frontiers=True)
    assert not obs.frontier_summary.any()
