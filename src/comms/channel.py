# A minimal in-memory message channel with a pluggable transport policy.
#
# Delivery is still immediate (same tick). A LinkModel decides *whether* each
# message reaches each recipient: uniform Bernoulli drop plus a per-recipient,
# per-tick bandwidth cap. A default LinkModel is a perfect (lossless, unlimited)
# link, so callers that don't care about realism see the original behavior.
#
# TODO: distance/range-based loss, then latency (delivery-tick gating here).

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from comms.link import LinkModel
from comms.message import Cell, Message


class CommsChannel:
    def __init__(self, link: LinkModel | None = None) -> None:
        # receiver_id -> messages waiting to be received
        self._inboxes: dict[str, list[Message]] = defaultdict(list)
        self._next_id = 0
        self._link = link if link is not None else LinkModel()

        # Per-recipient bytes delivered during the current tick, for the
        # bandwidth cap. Assumes ticks passed to send() are monotonic (the env
        # drives them that way); a new tick resets the accounting.
        self._tick: int | None = None
        self._tick_bytes: dict[str, int] = defaultdict(int)

        # Cumulative transport stats (a future comms metric reads these).
        self.bytes_delivered = 0
        self.bytes_dropped = 0
        self.messages_dropped = 0

    def reset(self, seed: int | None = None) -> None:
        self._inboxes = defaultdict(list)
        self._next_id = 0
        self._tick = None
        self._tick_bytes = defaultdict(int)
        self.bytes_delivered = 0
        self.bytes_dropped = 0
        self.messages_dropped = 0
        self._link.reset(seed)

    def send(
        self,
        sender_id: str,
        cells: tuple[Cell, ...],
        recipients: Iterable[str],
        tick: int,
    ) -> Message:
        # Queue a belief patch from sender_id for each recipient that the link
        # actually delivers to. Returns the constructed Message regardless of
        # per-recipient delivery (it exists on the wire even if it's dropped).
        message = Message.from_belief_delta(
            sender_id=sender_id,
            cells=cells,
            tick=tick,
            message_id=self._next_id,
        )
        self._next_id += 1

        # A new tick resets the per-recipient bandwidth accounting.
        if tick != self._tick:
            self._tick = tick
            self._tick_bytes = defaultdict(int)

        cap = self._link.max_bytes_per_tick
        for rid in recipients:
            # Uniform random loss.
            if self._link.should_drop():
                self.bytes_dropped += message.size_bytes
                self.messages_dropped += 1
                continue
            # Bandwidth cap: drop what doesn't fit this recipient's tick budget
            # (no deferral -- latency is a later stage).
            if cap is not None and self._tick_bytes[rid] + message.size_bytes > cap:
                self.bytes_dropped += message.size_bytes
                self.messages_dropped += 1
                continue
            self._inboxes[rid].append(message)
            self._tick_bytes[rid] += message.size_bytes
            self.bytes_delivered += message.size_bytes
        return message

    def receive(self, robot_id: str) -> list[Message]:
        # Pop and return every message queued for robot_id
        messages = self._inboxes.get(robot_id, [])
        self._inboxes[robot_id] = []
        return messages
