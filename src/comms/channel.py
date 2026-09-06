# A minimal in-memory message channel.
# TODO: Add distance/map based loss and latency

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from comms.message import Cell, Message


class CommsChannel:
    def __init__(self) -> None:
        # receiver_id -> messages waiting to be received
        self._inboxes: dict[str, list[Message]] = defaultdict(list)
        self._next_id = 0

    def reset(self) -> None:
        self._inboxes = defaultdict(list)
        self._next_id = 0

    def send(
        self,
        sender_id: str,
        cells: tuple[Cell, ...],
        recipients: Iterable[str],
        tick: int,
    ) -> Message:
        # Queue a belief patch from sender_id for each recipient.
        # Returns the constructed Message (the same object is delivered to every
        # recipient's inbox)
        message = Message.from_belief_delta(
            sender_id=sender_id,
            cells=cells,
            tick=tick,
            message_id=self._next_id,
        )
        self._next_id += 1
        for rid in recipients:
            self._inboxes[rid].append(message)
        return message

    def receive(self, robot_id: str) -> list[Message]:
        # Pop and return every message queued for robot_id
        messages = self._inboxes.get(robot_id, [])
        self._inboxes[robot_id] = []
        return messages
