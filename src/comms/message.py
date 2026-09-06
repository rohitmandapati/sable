# One message a robot sends across the (simulated) network, serialized as a sequence of packets

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from robot import KNOWN_FREE, KNOWN_WALL, Position

# Belief-patch cell: ((row, col), value) where value is KNOWN_FREE or KNOWN_WALL
Cell = tuple[Position, int]

# Serialized as int16 rows of [row, col, value]; int16 comfortably covers grid
# coordinates and the {0, 1} belief values while keeping packets compact
_PACKET_DTYPE = np.int16
_PACKET_COLS = 3


@dataclass
class Message:
    sender_id: str
    cells: tuple[Cell, ...] # cells the sender is sharing
    created_tick: int # Simulation tick the message was created on (metadata / ordering)
    message_id: int 

    # Derived on-the-wire representation and its size in bytes
    packets: bytes = field(init=False, repr=False)
    size_bytes: int = field(init=False)

    def __post_init__(self) -> None:
        self.cells = tuple(
            ((int(r), int(c)), int(v)) for (r, c), v in self.cells
        )
        for (_r, _c), v in self.cells:
            if v not in (KNOWN_FREE, KNOWN_WALL):
                raise ValueError(
                    f"Message cells must be KNOWN_FREE/KNOWN_WALL, got {v}"
                )
        self.packets = self._serialize(self.cells)
        self.size_bytes = len(self.packets)

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
            return np.empty((0, _PACKET_COLS), dtype=_PACKET_DTYPE).tobytes()
        rows = np.array([(r, c, v) for (r, c), v in cells], dtype=_PACKET_DTYPE)
        return rows.tobytes()

    @staticmethod
    def deserialize(packets: bytes) -> tuple[Cell, ...]:
        # Recover the belief patch from an on-the-wire packet blob
        rows = np.frombuffer(packets, dtype=_PACKET_DTYPE)
        if rows.size == 0:
            return ()
        rows = rows.reshape(-1, _PACKET_COLS)
        return tuple(((int(r), int(c)), int(v)) for r, c, v in rows)
