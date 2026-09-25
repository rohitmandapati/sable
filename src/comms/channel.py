# A minimal in-memory message channel with a pluggable transport policy.
#
# All-to-all with scheduled delivery: for each recipient the channel asks its
# LinkModel `evaluate(query) -> outcome`. The outcome decides delivery (drop
# physics) and a latency `delay_ticks`; a delivered message is queued and only
# handed to the recipient at send-tick + delay. The per-recipient per-tick
# bandwidth cap is applied at DELIVERY time (when a receiver actually ingests the
# tick's arrivals), not at send time.
#
# TODO: a more realistic transport stage would not be all-to-all (topology/routing).

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping

from comms.link import LinkModel, LinkQuery
from comms.message import Cell, Message
from robot import Position


class CommsChannel:
    def __init__(self, link: LinkModel | None = None) -> None:
        # receiver_id -> queued (deliver_at_tick, message), awaiting their tick.
        self._inboxes: dict[str, list[tuple[int, Message]]] = defaultdict(list)
        self._next_id = 0
        self._link = link if link is not None else LinkModel()

        # Cumulative payload-only transport stats. Payload = the belief cells
        # actually serialized on the wire (see Message); no metadata is counted.
        #   transmitted: counted once per logical broadcast (one send()).
        #   delivered: counted per recipient at delivery time (in receive()).
        #   dropped: link drops at send time, bandwidth drops at delivery time.
        self.payload_bytes_transmitted = 0
        self.payload_bytes_delivered = 0
        self.payload_bytes_dropped = 0
        self.deliveries_made = 0
        self.deliveries_dropped = 0
        # Per-cause non-delivery counts (out_of_range / occluded / stochastic
        # from the link, plus "bandwidth" for cap rejections here).
        self.drops_by_cause: dict[str, int] = defaultdict(int)
        # Latency stats over delivered messages (delay = deliver_at - created_tick).
        self.delay_sum = 0
        self.delay_max = 0
        self.delay_count = 0

    def reset(self, seed: int | None = None) -> None:
        self._inboxes = defaultdict(list)
        self._next_id = 0
        self.payload_bytes_transmitted = 0
        self.payload_bytes_delivered = 0
        self.payload_bytes_dropped = 0
        self.deliveries_made = 0
        self.deliveries_dropped = 0
        self.drops_by_cause = defaultdict(int)
        self.delay_sum = 0
        self.delay_max = 0
        self.delay_count = 0
        self._link.reset(seed)

    @property
    def mean_delay(self) -> float:
        # Average delivery latency (ticks) over delivered messages; 0 if none.
        return self.delay_sum / self.delay_count if self.delay_count else 0.0

    def messages_in_flight(self) -> int:
        # Messages accepted by the link but not yet drained (still latent).
        return sum(len(q) for q in self._inboxes.values())

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

        pos = positions or {}
        sender_pos = pos.get(sender_id)
        for rid in recipients:
            # Transport decision per recipient: delivery (drop physics) + latency.
            outcome = self._link.evaluate(
                LinkQuery(
                    sender_id=sender_id,
                    recipient_id=rid,
                    sender_pos=sender_pos,
                    recipient_pos=pos.get(rid),
                    tick=tick,
                    size_bytes=size,
                    message_id=message.message_id,
                )
            )
            if not outcome.delivered:
                self.payload_bytes_dropped += size
                self.deliveries_dropped += 1
                self.drops_by_cause[outcome.drop_cause or "link"] += 1
                continue
            # Delivered: queue for its arrival tick. The bandwidth cap is applied
            # when the recipient drains this tick's arrivals (see receive()).
            deliver_at = tick + max(0, outcome.delay_ticks)
            self._inboxes[rid].append((deliver_at, message))
        return message

    def receive(self, robot_id: str, tick: int | None = None) -> list[Message]:
        # Hand the recipient every message whose arrival tick has come (<= tick),
        # leaving still-latent messages queued. `tick=None` drains all queued
        # messages regardless of schedule (convenience for latency-free callers;
        # the env always passes the current tick). The per-recipient per-tick
        # bandwidth cap is applied here, oldest-first; overflow is dropped (no
        # deferral -- a message either lands on its arrival tick or not at all).
        inbox = self._inboxes.get(robot_id)
        if not inbox:
            return []
        if tick is None:
            ready, latent = inbox, []
        else:
            ready = [(t, m) for (t, m) in inbox if t <= tick]
            latent = [(t, m) for (t, m) in inbox if t > tick]
        self._inboxes[robot_id] = latent

        # Deterministic, order-independent tie-break: oldest message first.
        ready.sort(key=lambda tm: (tm[1].created_tick, tm[1].message_id))
        cap = self._link.max_bytes_per_tick
        used = 0
        delivered: list[Message] = []
        for deliver_at, message in ready:
            size = message.payload_size_bytes
            if cap is not None and used + size > cap:
                self.payload_bytes_dropped += size
                self.deliveries_dropped += 1
                self.drops_by_cause["bandwidth"] += 1
                continue
            used += size
            delivered.append(message)
            self.payload_bytes_delivered += size
            self.deliveries_made += 1
            delay = deliver_at - message.created_tick
            self.delay_sum += delay
            self.delay_count += 1
            self.delay_max = max(self.delay_max, delay)
        return delivered
