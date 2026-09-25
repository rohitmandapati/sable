# What a robot's *learned* comms head emits each tick.
#
# A CommunicationAction is the policy-facing description of one communication
# decision: whether to send at all, to whom, what kind of payload, and the
# payload itself. It is intentionally decoupled from any transport detail --
# packet loss, latency, bandwidth, distance, spoofing all live behind the
# CommsBackend. The action names an *endpoint* (broadcast / unicast / skip); the
# backend decides what actually gets delivered.
#
# First cut deliberately supports only send/skip + broadcast-vs-unicast with a
# belief-delta payload. Learned latent payloads and variable message sizes are
# reserved (see PayloadKind) and slot in without changing this shape.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from comms.message import Cell


class PayloadKind(Enum):
    # What the payload cells mean. Only BELIEF_DELTA is carried on the wire today
    # (it reuses Message's cell serialization); the rest are reserved so the
    # sender head can grow message *types* without reworking the action.
    BELIEF_DELTA = "belief_delta"      # newly-known cells since last send
    FRONTIER_SUMMARY = "frontier_summary"  # reserved: compact frontier digest
    POSITION_INTENT = "position_intent"    # reserved: pose + intended goal
    LATENT = "latent"                      # reserved: learned message embedding


@dataclass(frozen=True)
class CommunicationAction:
    # send=False is "skip" -- the policy chose to spend no bandwidth this tick.
    send: bool = False
    # None => broadcast to every reachable peer (the backend supplies the
    # roster). A concrete tuple => unicast to exactly those recipient ids.
    recipients: tuple[str, ...] | None = None
    payload_kind: PayloadKind = PayloadKind.BELIEF_DELTA
    # The belief patch being shared. Empty is legal (a send with nothing new).
    cells: tuple[Cell, ...] = ()

    @property
    def is_broadcast(self) -> bool:
        return self.send and self.recipients is None

    # -- convenience constructors (one per backend endpoint) -----------------

    @classmethod
    def skip(cls) -> "CommunicationAction":
        return cls(send=False)

    @classmethod
    def broadcast(
        cls,
        cells: tuple[Cell, ...],
        payload_kind: PayloadKind = PayloadKind.BELIEF_DELTA,
    ) -> "CommunicationAction":
        return cls(send=True, recipients=None, payload_kind=payload_kind, cells=tuple(cells))

    @classmethod
    def unicast(
        cls,
        recipients: tuple[str, ...],
        cells: tuple[Cell, ...],
        payload_kind: PayloadKind = PayloadKind.BELIEF_DELTA,
    ) -> "CommunicationAction":
        return cls(
            send=True,
            recipients=tuple(recipients),
            payload_kind=payload_kind,
            cells=tuple(cells),
        )
