# The communication API the learned policy talks to -- and nothing more.
#
# A CommsBackend is the ONLY comms surface the policy sees. The policy emits a
# CommunicationAction; the backend decides what is actually delivered and when.
# Every realism concern (loss, latency, bandwidth, distance, partitions,
# spoofing) is a property of a *backend implementation*, never of the action or
# the policy. Swapping backends must not change the policy -- that invariant is
# the whole point of this seam.
#
# This module ships the first, trivial backend: PerfectBroadcastBackend --
# immediate, lossless, unlimited delivery to every peer. It exists so the MAPPO
# pipeline can run end-to-end before any realism is modeled. Because the protocol
# already expresses "delivery MAY be partial or delayed" (deliver() can return
# fewer messages than were submitted, with age >= 1), the degraded backends drop
# in later behind this same interface with zero policy change.
#
# Timing convention: submit()/execute() only ENQUEUE; deliver(recipient, tick)
# drains that recipient's inbox and stamps delivered_tick=tick. The environment
# drains at the start of a tick, so a message submitted at tick t is delivered at
# t+1 (age 1) under this backend.

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from comms.action import CommunicationAction, PayloadKind
from comms.delivered import DeliveredMessage
from comms.message import Cell, Message
from robot import Position


@dataclass
class CommsStats:
    # Payload-only accounting (metadata is not on the wire; see Message). Counts
    # are cumulative for the episode and reset between episodes.
    messages_transmitted: int = 0   # one per logical send, regardless of fan-out
    deliveries_made: int = 0        # one per (message, recipient) actually landed
    payload_bytes_transmitted: int = 0
    payload_bytes_delivered: int = 0

    def reset(self) -> None:
        self.messages_transmitted = 0
        self.deliveries_made = 0
        self.payload_bytes_transmitted = 0
        self.payload_bytes_delivered = 0


@runtime_checkable
class CommsBackend(Protocol):
    # The endpoints a robot's comms action can invoke. `execute` routes a whole
    # CommunicationAction; `broadcast`/`send_to` are the raw endpoints it maps to.
    stats: CommsStats

    def reset(self, seed: int | None = None) -> None: ...

    def execute(
        self,
        action: CommunicationAction,
        *,
        sender_id: str,
        tick: int,
        sender_position: Position | None = None,
    ) -> Message | None: ...

    def deliver(self, recipient_id: str, tick: int) -> list[DeliveredMessage]: ...


# One queued item: the message as reconstructed from its wire frame, plus the
# delivery metadata we still carry out-of-band (claimed position, payload kind)
# until a backend chooses to model those on the wire too.
_Pending = tuple[Message, Position | None, PayloadKind]


class PerfectBroadcastBackend:
    """Immediate, lossless, unlimited-bandwidth delivery to every peer."""

    def __init__(self, robot_ids: Iterable[str]) -> None:
        self._ids: list[str] = list(robot_ids)
        self._inboxes: dict[str, list[_Pending]] = defaultdict(list)
        self._next_id = 0
        self.stats = CommsStats()

    def reset(self, seed: int | None = None) -> None:
        # seed is accepted for protocol symmetry; a lossless backend has no RNG.
        self._inboxes = defaultdict(list)
        self._next_id = 0
        self.stats.reset()

    # -- endpoints -----------------------------------------------------------

    def execute(
        self,
        action: CommunicationAction,
        *,
        sender_id: str,
        tick: int,
        sender_position: Position | None = None,
    ) -> Message | None:
        # Route one CommunicationAction to the matching endpoint. Skip is a no-op.
        if not action.send:
            return None
        recipients = self._recipients(sender_id, action.recipients)
        return self._transmit(sender_id, sender_position, action, recipients, tick)

    def broadcast(
        self,
        sender_id: str,
        cells: tuple[Cell, ...],
        *,
        tick: int,
        sender_position: Position | None = None,
        payload_kind: PayloadKind = PayloadKind.BELIEF_DELTA,
    ) -> Message:
        action = CommunicationAction.broadcast(cells, payload_kind)
        return self._transmit(
            sender_id, sender_position, action, self._recipients(sender_id, None), tick
        )

    def send_to(
        self,
        sender_id: str,
        recipients: Iterable[str],
        cells: tuple[Cell, ...],
        *,
        tick: int,
        sender_position: Position | None = None,
        payload_kind: PayloadKind = PayloadKind.BELIEF_DELTA,
    ) -> Message:
        action = CommunicationAction.unicast(tuple(recipients), cells, payload_kind)
        return self._transmit(
            sender_id, sender_position, action, self._recipients(sender_id, action.recipients), tick
        )

    def deliver(self, recipient_id: str, tick: int) -> list[DeliveredMessage]:
        # Drain this recipient's inbox. Each pending item was already
        # reconstructed from its wire frame at submit time, so the receiver reads
        # sender/id/tick/cells off the wire -- never the sender's live object.
        pending = self._inboxes.get(recipient_id, [])
        self._inboxes[recipient_id] = []
        out: list[DeliveredMessage] = []
        for received, sender_position, kind in pending:
            self.stats.deliveries_made += 1
            self.stats.payload_bytes_delivered += received.payload_size_bytes
            out.append(
                DeliveredMessage(
                    sender_id=received.sender_id,
                    payload_kind=kind,
                    cells=received.cells,
                    created_tick=received.created_tick,
                    delivered_tick=tick,
                    sequence_id=received.message_id,
                    payload_size_bytes=received.payload_size_bytes,
                    sender_position=sender_position,
                )
            )
        return out

    # -- internals -----------------------------------------------------------

    def _recipients(
        self, sender_id: str, recipients: tuple[str, ...] | None
    ) -> list[str]:
        # None => broadcast to the whole roster; a sender never messages itself.
        pool = self._ids if recipients is None else recipients
        return [rid for rid in pool if rid != sender_id]

    def _transmit(
        self,
        sender_id: str,
        sender_position: Position | None,
        action: CommunicationAction,
        recipients: list[str],
        tick: int,
    ) -> Message:
        # Build the wire Message once (serialize), then reconstruct it from its
        # full frame so recipients receive exactly what survived the wire --
        # provenance included -- the honest path, even though this backend never
        # corrupts it.
        message = Message.from_belief_delta(
            sender_id=sender_id,
            cells=action.cells,
            tick=tick,
            message_id=self._next_id,
        )
        self._next_id += 1
        self.stats.messages_transmitted += 1
        self.stats.payload_bytes_transmitted += message.payload_size_bytes

        received = Message.deserialize(message.wire_bytes)
        for rid in recipients:
            self._inboxes[rid].append((received, sender_position, action.payload_kind))
        return message
