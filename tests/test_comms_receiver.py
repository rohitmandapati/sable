"""Unit tests for the receive/trust side (ReceiveInbox + ReceiverTrustPolicy +
per-cell trust fusion). conftest.py puts src/ on sys.path, so imports are flat.

These pin the contract the learned trust head slots into: delivery enqueues,
interpretation decides + trusts, fusion writes belief AND the trust overlay
without ever touching first-hand sensing.
"""

import numpy as np
import pytest

from comms import (
    PayloadKind,
    PerfectBroadcastBackend,
    ReceiveContext,
    ReceiveInbox,
    ReceiverAction,
    ReceiverDecision,
    TrustAllReceiver,
    process_inbox,
)
from comms.delivered import DeliveredMessage
from robot import KNOWN_FREE, KNOWN_WALL, UNKNOWN, Robot


def _delivered(cells, *, sender="a", created=1, delivered=2, seq=0):
    return DeliveredMessage(
        sender_id=sender,
        payload_kind=PayloadKind.BELIEF_DELTA,
        cells=tuple(cells),
        created_tick=created,
        delivered_tick=delivered,
        sequence_id=seq,
        payload_size_bytes=len(cells) * 6,
    )


# -- ReceiverAction -------------------------------------------------------------

def test_trust_must_be_a_probability():
    ReceiverAction(ReceiverDecision.FUSE, trust=0.0)
    ReceiverAction(ReceiverDecision.FUSE, trust=1.0)
    with pytest.raises(ValueError):
        ReceiverAction(ReceiverDecision.FUSE, trust=1.5)


# -- ReceiveInbox ---------------------------------------------------------------

def test_inbox_enqueues_and_drains_in_order():
    inbox = ReceiveInbox()
    inbox.enqueue([_delivered([((0, 0), KNOWN_FREE)], seq=0)])
    inbox.enqueue([_delivered([((0, 1), KNOWN_FREE)], seq=1)])
    assert len(inbox) == 2
    drained = inbox.drain()
    assert [m.sequence_id for m in drained] == [0, 1]
    assert len(inbox) == 0          # drain clears
    assert inbox.drain() == []


# -- fusion into belief + trust overlay ----------------------------------------

def test_fuse_writes_belief_and_trust_but_not_sensed_mask():
    robot = Robot(robot_id="r", pos=(0, 0), map_shape=(4, 4))
    inbox = ReceiveInbox()
    inbox.enqueue([_delivered([((2, 3), KNOWN_WALL)])])
    relayed = process_inbox(robot, inbox, TrustAllReceiver(), tick=2)
    assert relayed == []
    assert robot.belief_map[2, 3] == KNOWN_WALL     # belief filled
    assert robot.trust_map[2, 3] == 1.0             # trust stamped
    assert not robot.sensed_mask[2, 3]              # first-hand record untouched


def test_fuse_does_not_overwrite_known_belief_but_still_records_trust():
    robot = Robot(robot_id="r", pos=(0, 0), map_shape=(4, 4))
    robot.belief_map[1, 1] = KNOWN_FREE             # already known first-hand
    inbox = ReceiveInbox()
    inbox.enqueue([_delivered([((1, 1), KNOWN_WALL)])])   # conflicting claim
    process_inbox(robot, inbox, TrustAllReceiver(), tick=2)
    assert robot.belief_map[1, 1] == KNOWN_FREE     # belief unchanged
    assert robot.trust_map[1, 1] == 1.0             # trust still recorded


def test_trust_accumulates_by_max():
    robot = Robot(robot_id="r", pos=(0, 0), map_shape=(4, 4))
    robot.fuse_cell((0, 2), KNOWN_FREE, trust=0.4)
    robot.fuse_cell((0, 2), KNOWN_FREE, trust=0.9)
    robot.fuse_cell((0, 2), KNOWN_FREE, trust=0.6)
    assert robot.trust_map[0, 2] == pytest.approx(0.9)


# -- decisions ------------------------------------------------------------------

class _FixedReceiver:
    def __init__(self, action):
        self._action = action

    def interpret(self, message, context):
        return self._action


def test_discard_leaves_belief_untouched():
    robot = Robot(robot_id="r", pos=(0, 0), map_shape=(4, 4))
    inbox = ReceiveInbox()
    inbox.enqueue([_delivered([((3, 3), KNOWN_FREE)])])
    policy = _FixedReceiver(ReceiverAction(ReceiverDecision.DISCARD))
    relayed = process_inbox(robot, inbox, policy, tick=2)
    assert relayed == []
    assert robot.belief_map[3, 3] == UNKNOWN
    assert robot.trust_map[3, 3] == 0.0


def test_relay_is_recorded_not_fused():
    robot = Robot(robot_id="r", pos=(0, 0), map_shape=(4, 4))
    inbox = ReceiveInbox()
    msg = _delivered([((3, 3), KNOWN_FREE)], seq=7)
    inbox.enqueue([msg])
    policy = _FixedReceiver(ReceiverAction(ReceiverDecision.RELAY, trust=0.5))
    relayed = process_inbox(robot, inbox, policy, tick=2)
    assert [m.sequence_id for m in relayed] == [7]  # surfaced for forwarding
    assert robot.belief_map[3, 3] == UNKNOWN        # but NOT added to belief
    assert robot.trust_map[3, 3] == 0.0


def test_partial_trust_is_stamped_on_fused_cells():
    robot = Robot(robot_id="r", pos=(0, 0), map_shape=(4, 4))
    inbox = ReceiveInbox()
    inbox.enqueue([_delivered([((0, 3), KNOWN_FREE)])])
    policy = _FixedReceiver(ReceiverAction(ReceiverDecision.FUSE, trust=0.25))
    process_inbox(robot, inbox, policy, tick=2)
    assert robot.belief_map[0, 3] == KNOWN_FREE
    assert robot.trust_map[0, 3] == pytest.approx(0.25)


# -- end to end through the backend --------------------------------------------

def test_backend_delivery_flows_into_inbox_and_fuses():
    backend = PerfectBroadcastBackend(["a", "b"])
    robot_b = Robot(robot_id="b", pos=(0, 0), map_shape=(4, 4))
    inbox = ReceiveInbox()

    backend.broadcast("a", (((1, 2), KNOWN_FREE),), tick=1)
    inbox.enqueue(backend.deliver("b", tick=2))
    assert len(inbox) == 1

    process_inbox(robot_b, inbox, TrustAllReceiver(), tick=2)
    assert robot_b.belief_map[1, 2] == KNOWN_FREE
    assert robot_b.trust_map[1, 2] == 1.0
