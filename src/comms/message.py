# One message a robot broadcasts, whose on-the-wire form is *only* the belief
# payload -- the cells being shared. Sender id, timestamps, and message ids are
# kept as in-memory metadata and are deliberately NOT serialized yet; the byte
# accounting must stay honest about what is actually on the wire.

from __future__ import annotations

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


@dataclass
class Message:
    sender_id: str
    cells: tuple[Cell, ...] # cells the sender is sharing
    created_tick: int # Simulation tick the message was created on (metadata / ordering)
    message_id: int

    # Derived on-the-wire representation (belief payload only) and its size.
    payload_bytes: bytes = field(init=False, repr=False)
    payload_size_bytes: int = field(init=False)

    def __post_init__(self) -> None:
        self.cells = tuple(
            ((int(r), int(c)), int(v)) for (r, c), v in self.cells
        )
        for (_r, _c), v in self.cells:
            if v not in (KNOWN_FREE, KNOWN_WALL):
                raise ValueError(
                    f"Message cells must be KNOWN_FREE/KNOWN_WALL, got {v}"
                )
        self.payload_bytes = self._serialize(self.cells)
        self.payload_size_bytes = len(self.payload_bytes)

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
