# One message a robot broadcasts. Its on-the-wire form is a serialized HEADER
# (sender id, message id, created tick) followed by the belief PAYLOAD (the cells
# being shared), so a receiver reconstructs the whole message -- provenance and
# all -- from bytes alone. `wire_bytes` is the full frame; `payload_bytes` is
# just the payload, kept separate so cost accounting can distinguish novel
# information from framing overhead.

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

from robot import KNOWN_FREE, KNOWN_WALL, Position

# Belief-patch cell: ((row, col), value) where value is KNOWN_FREE or KNOWN_WALL
Cell = tuple[Position, int]

# Payload wire format: each cell is exactly three explicit little-endian int16
# values -- row, column, belief value -- so one cell is 6 bytes. int16 (LE)
# comfortably covers grid coordinates and the {0, 1} belief values while keeping
# the payload compact and byte-order deterministic across machines.
_PAYLOAD_DTYPE = np.dtype("<i2")
_PAYLOAD_COLS = 3
_BYTES_PER_CELL = _PAYLOAD_COLS * _PAYLOAD_DTYPE.itemsize  # 6

# Header wire format (little-endian): a uint16 sender-id byte length, then that
# many UTF-8 bytes, then two int32s -- message id and created tick. int32 (not
# the payload's int16) because message ids and ticks routinely exceed 32767 over
# an episode. The payload is simply the remainder of the frame after the header.
_ID_LEN_FMT = "<H"
_ID_LEN_SIZE = struct.calcsize(_ID_LEN_FMT)  # 2
_META_FMT = "<ii"
_META_SIZE = struct.calcsize(_META_FMT)      # 8
_INT32_MAX = 2**31 - 1


@dataclass
class Message:
    sender_id: str
    cells: tuple[Cell, ...] # cells the sender is sharing
    created_tick: int # Simulation tick the message was created on
    message_id: int

    # Derived on-the-wire representations: the belief payload alone, and the full
    # header+payload frame, each with its size in bytes.
    payload_bytes: bytes = field(init=False, repr=False)
    payload_size_bytes: int = field(init=False)
    wire_bytes: bytes = field(init=False, repr=False)
    wire_size_bytes: int = field(init=False)

    def __post_init__(self) -> None:
        self.cells = tuple(
            ((int(r), int(c)), int(v)) for (r, c), v in self.cells
        )
        for (_r, _c), v in self.cells:
            if v not in (KNOWN_FREE, KNOWN_WALL):
                raise ValueError(
                    f"Message cells must be KNOWN_FREE/KNOWN_WALL, got {v}"
                )
        self.message_id = int(self.message_id)
        self.created_tick = int(self.created_tick)
        self.payload_bytes = self._serialize(self.cells)
        self.payload_size_bytes = len(self.payload_bytes)
        self.wire_bytes = (
            self._serialize_header(self.sender_id, self.message_id, self.created_tick)
            + self.payload_bytes
        )
        self.wire_size_bytes = len(self.wire_bytes)

    @classmethod
    def from_belief_delta(
        cls,
        sender_id: str,
        cells: tuple[Cell, ...],
        tick: int,
        message_id: int,
    ) -> "Message":
        # Build a message from a batch of known belief cells
        return cls(
            sender_id=sender_id,
            cells=tuple(cells),
            created_tick=tick,
            message_id=message_id,
        )

    @staticmethod
    def _serialize(cells: tuple[Cell, ...]) -> bytes:
        if not cells:
            return np.empty((0, _PAYLOAD_COLS), dtype=_PAYLOAD_DTYPE).tobytes()
        rows = np.array([(r, c, v) for (r, c), v in cells], dtype=_PAYLOAD_DTYPE)
        return rows.tobytes()

    @staticmethod
    def deserialize_cells(payload_bytes: bytes) -> tuple[Cell, ...]:
        # Recover the belief patch from a payload blob. Each cell is three
        # little-endian int16 values (row, col, belief), so a well-formed payload
        # is a whole number of 6-byte records carrying only {0, 1} belief values.
        if len(payload_bytes) % _BYTES_PER_CELL != 0:
            raise ValueError(
                f"malformed payload: {len(payload_bytes)} bytes is not a multiple "
                f"of {_BYTES_PER_CELL} (row,col,value as little-endian int16)"
            )
        rows = np.frombuffer(payload_bytes, dtype=_PAYLOAD_DTYPE)
        if rows.size == 0:
            return ()
        rows = rows.reshape(-1, _PAYLOAD_COLS)
        cells: list[Cell] = []
        for r, c, v in rows:
            v = int(v)
            if v not in (KNOWN_FREE, KNOWN_WALL):
                raise ValueError(
                    f"malformed payload: decoded belief value {v} is not "
                    f"KNOWN_FREE/KNOWN_WALL"
                )
            cells.append(((int(r), int(c)), v))
        return tuple(cells)

    @staticmethod
    def _serialize_header(sender_id: str, message_id: int, created_tick: int) -> bytes:
        # uint16 sender-id length + UTF-8 sender id + int32 message_id + int32
        # created_tick. Ranges are checked so a value that can't be framed fails
        # loudly here rather than silently truncating on the wire.
        sender = sender_id.encode("utf-8")
        if len(sender) > 0xFFFF:
            raise ValueError(f"sender_id too long to frame: {len(sender)} bytes")
        if not (0 <= message_id <= _INT32_MAX):
            raise ValueError(f"message_id out of int32 range: {message_id}")
        if not (0 <= created_tick <= _INT32_MAX):
            raise ValueError(f"created_tick out of int32 range: {created_tick}")
        return (
            struct.pack(_ID_LEN_FMT, len(sender))
            + sender
            + struct.pack(_META_FMT, message_id, created_tick)
        )

    @classmethod
    def deserialize(cls, wire_bytes: bytes) -> "Message":
        # Recover a full Message -- provenance and payload -- from its wire frame.
        # The payload is whatever remains after the header.
        if len(wire_bytes) < _ID_LEN_SIZE:
            raise ValueError("malformed frame: too short for a sender-id length")
        (id_len,) = struct.unpack_from(_ID_LEN_FMT, wire_bytes, 0)
        off = _ID_LEN_SIZE
        if len(wire_bytes) < off + id_len + _META_SIZE:
            raise ValueError("malformed frame: truncated header")
        sender_id = wire_bytes[off : off + id_len].decode("utf-8")
        off += id_len
        message_id, created_tick = struct.unpack_from(_META_FMT, wire_bytes, off)
        off += _META_SIZE
        cells = cls.deserialize_cells(wire_bytes[off:])
        return cls(
            sender_id=sender_id,
            cells=cells,
            created_tick=created_tick,
            message_id=message_id,
        )
