# A minimal in-memory message channel with a pluggable transport policy.
#
# Delivery is still immediate (same tick) and all-to-all: for each recipient the
# channel asks its LinkModel `evaluate(query) -> outcome` (uniform Bernoulli drop
# today) and then applies a per-recipient, per-tick bandwidth cap.
#
# TODO: a more realistic transport stage would queue messages for later delivery
# and not be all-to-all

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping

from comms.link import LinkModel, LinkQuery
from comms.message import Cell, Message
from robot import Position


class CommsChannel:
    def __init__(self, link: LinkModel | None = None) -> None:
        # receiver_id -> messages waiting to be received
        self._inboxes: dict[str, list[Message]] = defaultdict(list)
        self._next_id = 0
        self._link = link if link is not None else LinkModel()

        # Per-recipient bytes delivered during the current tick, for the
        # bandwidth cap. Assumes ticks passed to send() are monotonic (the env
        # drives them that way); a new tick resets the accounting. 
        # NOTE: max_bytes_per_tick is currently a per-recipient delivered-payload
        # limit (it caps the belief-payload bytes each recipient accepts per
        # tick), this is not a redesign of the bandwidth model.
        self._tick: int | None = None
        self._tick_bytes: dict[str, int] = defaultdict(int)

        # Cumulative payload-only transport stats. Payload = the belief cells
        # actually serialized on the wire (see Message); no metadata is counted.
        #   transmitted: counted once per logical broadcast (one send()).
        #   delivered/dropped: counted once per recipient outcome.
        self.payload_bytes_transmitted = 0
        self.payload_bytes_delivered = 0
        self.payload_bytes_dropped = 0
        self.deliveries_made = 0
        self.deliveries_dropped = 0
        # Per-cause non-delivery counts (out_of_range / occluded / stochastic
        # from the link, plus "bandwidth" for cap rejections here).
        self.drops_by_cause: dict[str, int] = defaultdict(int)

    def reset(self, seed: int | None = None) -> None:
        self._inboxes = defaultdict(list)
        self._next_id = 0
        self._tick = None
        self._tick_bytes = defaultdict(int)
        self.payload_bytes_transmitted = 0
        self.payload_bytes_delivered = 0
        self.payload_bytes_dropped = 0
        self.deliveries_made = 0
        self.deliveries_dropped = 0
        self.drops_by_cause = defaultdict(int)
        self._link.reset(seed)

    def send(
        self,
        sender_id: str,
        cells: tuple[Cell, ...],
        recipients: Iterable[str],
        tick: int,
        positions: Mapping[str, Position] | None = None,
    ) -> Message:
        # Queue a belief patch from sender_id for each recipient that the link
        # actually delivers to. Returns the constructed Message regardless of
        # per-recipient delivery (it exists on the wire even if it's dropped).
        # `positions` (robot_id -> cell) is passed to the link for distance-aware
        # stages; the uniform link ignores it.
        message = Message.from_belief_delta(
            sender_id=sender_id,
            cells=cells,
            tick=tick,
            message_id=self._next_id,
        )
        self._next_id += 1
        size = message.payload_size_bytes

        # One logical broadcast, transmitted once regardless of recipient count.
        self.payload_bytes_transmitted += size

        # A new tick resets the per-recipient bandwidth accounting.
        if tick != self._tick:
            self._tick = tick
            self._tick_bytes = defaultdict(int)

        pos = positions or {}
        sender_pos = pos.get(sender_id)
        cap = self._link.max_bytes_per_tick
        for rid in recipients:
            # Transport decision (uniform random loss today). Evaluated per
            # recipient, before the bandwidth check, exactly as before.
            outcome = self._link.evaluate(
                LinkQuery(
                    sender_id=sender_id,
                    recipient_id=rid,
                    sender_pos=sender_pos,
                    recipient_pos=pos.get(rid),
                    tick=tick,
                    size_bytes=size,
                )
            )
            if not outcome.delivered:
                self.payload_bytes_dropped += size
                self.deliveries_dropped += 1
                self.drops_by_cause[outcome.drop_cause or "link"] += 1
                continue
            # Bandwidth cap: drop what doesn't fit this recipient's per-tick
            # delivered-payload budget (no deferral -- latency is a later stage).
            if cap is not None and self._tick_bytes[rid] + size > cap:
                self.payload_bytes_dropped += size
                self.deliveries_dropped += 1
                self.drops_by_cause["bandwidth"] += 1
                continue
            self._inboxes[rid].append(message)
            self._tick_bytes[rid] += size
            self.payload_bytes_delivered += size
            self.deliveries_made += 1
        return message

    def receive(self, robot_id: str) -> list[Message]:
        # Pop and return every message queued for robot_id
        messages = self._inboxes.get(robot_id, [])
        self._inboxes[robot_id] = []
        return messages
