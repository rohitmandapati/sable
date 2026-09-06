"""Unit tests for the basic comms slice (Message + CommsChannel).

conftest.py puts src/ on sys.path, so imports are flat (from comms import ...).
"""

import pytest

from comms import CommsChannel, LinkModel, Message
from robot import KNOWN_FREE, KNOWN_WALL


def _cells(n):
    # n distinct free cells along a row.
    return tuple(((0, c), KNOWN_FREE) for c in range(n))


# -- Message --------------------------------------------------------------------

def test_message_round_trips_through_packets():
    cells = (((1, 2), KNOWN_FREE), ((3, 4), KNOWN_WALL))
    m = Message.from_belief_delta("r0", cells, tick=5, message_id=0)
    assert Message.deserialize(m.packets) == cells


def test_empty_patch_round_trips():
    m = Message.from_belief_delta("r0", (), tick=0, message_id=0)
    assert m.cells == ()
    assert Message.deserialize(m.packets) == ()
    assert m.size_bytes == 0


def test_size_bytes_scales_with_cell_count():
    small = Message.from_belief_delta("r0", _cells(1), tick=0, message_id=0)
    big = Message.from_belief_delta("r0", _cells(5), tick=0, message_id=1)
    assert big.size_bytes > small.size_bytes


def test_message_rejects_non_belief_values():
    with pytest.raises(ValueError):
        Message.from_belief_delta("r0", (((0, 0), -1),), tick=0, message_id=0)


# -- CommsChannel ---------------------------------------------------------------

def test_send_delivers_to_each_recipient():
    ch = CommsChannel()
    ch.send("r0", _cells(2), recipients=["r1", "r2"], tick=0)
    assert len(ch.receive("r1")) == 1
    assert len(ch.receive("r2")) == 1
    assert ch.receive("r3") == []  # nobody sent to r3


def test_receive_drains_the_inbox():
    ch = CommsChannel()
    ch.send("r0", _cells(1), recipients=["r1"], tick=0)
    assert len(ch.receive("r1")) == 1
    assert ch.receive("r1") == []  # already drained


def test_messages_queue_until_received():
    ch = CommsChannel()
    ch.send("r0", _cells(1), recipients=["r1"], tick=0)
    ch.send("r2", _cells(1), recipients=["r1"], tick=1)
    received = ch.receive("r1")
    assert [m.sender_id for m in received] == ["r0", "r2"]


def test_message_ids_are_unique():
    ch = CommsChannel()
    a = ch.send("r0", _cells(1), recipients=["r1"], tick=0)
    b = ch.send("r0", _cells(1), recipients=["r1"], tick=0)
    assert a.message_id != b.message_id


def test_reset_clears_inboxes():
    ch = CommsChannel()
    ch.send("r0", _cells(1), recipients=["r1"], tick=0)
    ch.reset()
    assert ch.receive("r1") == []
