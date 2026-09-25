"""Unit tests for the comms wire message (Message).

conftest.py puts src/ on sys.path, so imports are flat (from comms import ...).
"""

import numpy as np
import pytest

from comms import Message
from robot import KNOWN_FREE, KNOWN_WALL


def _cells(n):
    # n distinct free cells along a row.
    return tuple(((0, c), KNOWN_FREE) for c in range(n))


# -- Message --------------------------------------------------------------------

def test_message_round_trips_through_payload():
    cells = (((1, 2), KNOWN_FREE), ((3, 4), KNOWN_WALL))
    m = Message.from_belief_delta("r0", cells, tick=5, message_id=0)
    assert Message.deserialize_cells(m.payload_bytes) == cells


def test_message_wire_frame_round_trips_metadata_and_payload():
    # The full frame carries provenance too: a receiver recovers sender/id/tick
    # and the payload from bytes alone. The tick/id here exceed int16, proving
    # the int32 header.
    cells = (((1, 2), KNOWN_FREE), ((3, 4), KNOWN_WALL))
    m = Message.from_belief_delta("robot-7", cells, tick=40_000, message_id=99_999)
    back = Message.deserialize(m.wire_bytes)
    assert back.sender_id == "robot-7"
    assert back.created_tick == 40_000
    assert back.message_id == 99_999
    assert back.cells == cells


def test_wire_frame_is_larger_than_payload_by_the_header():
    m = Message.from_belief_delta("r0", (((0, 0), KNOWN_FREE),), tick=1, message_id=2)
    # header = 2 (len prefix) + 2 (len("r0")) + 8 (two int32) = 12 bytes.
    assert m.wire_size_bytes == m.payload_size_bytes + 12


def test_deserialize_rejects_truncated_header():
    # Claims a 5-byte sender id but only two bytes follow.
    with pytest.raises(ValueError):
        Message.deserialize(b"\x05\x00ab")


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


# Transport (send/deliver/stats) is covered by test_comms_backend.py against the
# CommsBackend seam; this module now only pins the wire Message format.
