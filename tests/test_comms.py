"""Unit tests for the basic comms slice (Message + CommsChannel).

conftest.py puts src/ on sys.path, so imports are flat (from comms import ...).
"""

import numpy as np
import pytest

from comms import CommsChannel, LinkModel, Message
from robot import KNOWN_FREE, KNOWN_WALL


def _cells(n):
    # n distinct free cells along a row.
    return tuple(((0, c), KNOWN_FREE) for c in range(n))


# -- Message --------------------------------------------------------------------

def test_message_round_trips_through_payload():
    cells = (((1, 2), KNOWN_FREE), ((3, 4), KNOWN_WALL))
    m = Message.from_belief_delta("r0", cells, tick=5, message_id=0)
    assert Message.deserialize_cells(m.payload_bytes) == cells


def test_empty_patch_round_trips():
    m = Message.from_belief_delta("r0", (), tick=0, message_id=0)
    assert m.cells == ()
    assert Message.deserialize_cells(m.payload_bytes) == ()
    assert m.payload_size_bytes == 0


def test_payload_size_scales_with_cell_count():
    small = Message.from_belief_delta("r0", _cells(1), tick=0, message_id=0)
    big = Message.from_belief_delta("r0", _cells(5), tick=0, message_id=1)
    assert big.payload_size_bytes > small.payload_size_bytes
    assert small.payload_size_bytes == 6  # one cell = three little-endian int16


def test_message_rejects_non_belief_values():
    with pytest.raises(ValueError):
        Message.from_belief_delta("r0", (((0, 0), -1),), tick=0, message_id=0)


def test_deserialize_rejects_malformed_length():
    # A payload must be a whole number of 6-byte (row, col, value) records.
    with pytest.raises(ValueError):
        Message.deserialize_cells(b"\x00\x00\x00")  # 3 bytes


def test_deserialize_rejects_invalid_belief_value():
    # Well-formed length, but the decoded belief value is neither free nor wall.
    blob = np.array([[0, 0, 5]], dtype=np.dtype("<i2")).tobytes()
    with pytest.raises(ValueError):
        Message.deserialize_cells(blob)


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


# -- payload-only transport counters --------------------------------------------

def test_payload_counters_transmitted_once_delivered_per_recipient():
    ch = CommsChannel()  # lossless, unlimited
    m = ch.send("r0", _cells(2), recipients=["r1", "r2", "r3"], tick=0)
    assert ch.payload_bytes_transmitted == m.payload_size_bytes  # once per broadcast
    assert ch.payload_bytes_delivered == 3 * m.payload_size_bytes  # per recipient
    assert ch.deliveries_made == 3
    assert ch.deliveries_dropped == 0
    assert ch.payload_bytes_dropped == 0


def test_payload_counters_count_drops():
    ch = CommsChannel(LinkModel(drop_prob=1.0))
    m = ch.send("r0", _cells(2), recipients=["r1", "r2"], tick=0)
    assert ch.payload_bytes_transmitted == m.payload_size_bytes  # still transmitted
    assert ch.payload_bytes_delivered == 0
    assert ch.deliveries_made == 0
    assert ch.deliveries_dropped == 2
    assert ch.payload_bytes_dropped == 2 * m.payload_size_bytes


def test_bandwidth_cap_is_per_recipient_delivered_payload_limit():
    one_cell = 6  # three little-endian int16
    ch = CommsChannel(LinkModel(max_bytes_per_tick=one_cell))
    ch.send("r0", _cells(1), recipients=["r1"], tick=0)  # fits the budget
    ch.send("r2", _cells(1), recipients=["r1"], tick=0)  # same tick -> over budget
    assert ch.deliveries_made == 1
    assert ch.deliveries_dropped == 1
    assert len(ch.receive("r1")) == 1
