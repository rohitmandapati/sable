# The RECEIVE / TRUST head -- the second learned policy.
#
# Delivery and interpretation are two separate steps. A CommsBackend DELIVERS
# messages into a robot's ReceiveInbox; nothing happens to belief yet. Then, per
# tick, the receive/trust policy INTERPRETS each queued message and emits a
# ReceiverAction: what to do with it (discard / fuse into belief / relay onward)
# and how much to trust it. `process_inbox` applies those actions -- FUSE writes
# the message's cells into belief with the chosen trust stamped onto the robot's
# per-cell trust overlay (Robot.trust_map); RELAY is recorded for forwarding but
# NOT re-injected yet; DISCARD drops it.
#
# The learned trust head will eventually score each message from features like
# staleness (DeliveredMessage.age), completeness, conflict-with-belief, and
# per-sender history. Those inputs flow in through ReceiveContext, which is
# deliberately small now and meant to grow. The shipped policy here is the
# trivial TrustAllReceiver baseline (fuse everything at full trust), so the
# pipeline runs before any trust model exists -- exactly how PerfectBroadcastBackend
# stands in for the transport.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Iterable, Protocol, runtime_checkable

import numpy as np

from comms.delivered import DeliveredMessage

if TYPE_CHECKING:  # avoid a hard import cycle; only needed for type hints
    from robot import Robot


class ReceiverDecision(Enum):
    DISCARD = "discard"  # throw the message out; belief untouched
    FUSE = "fuse"        # integrate the message's cells into belief + trust
    RELAY = "relay"      # forward onward (recorded only for now, not re-injected)


@dataclass(frozen=True)
class ReceiverAction:
    decision: ReceiverDecision
    # Confidence to stamp on fused cells, in [0, 1]. Only meaningful for FUSE,
    # but carried on every action so a relayed/discarded message still reports
    # the trust the head assigned it (useful for logging/rewards later).
    trust: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.trust <= 1.0:
            raise ValueError(f"trust must be in [0, 1], got {self.trust}")


@dataclass
class ReceiveContext:
    # What the trust head sees beyond the message itself. Small on purpose; the
    # learned head will grow this (per-sender history, latency/trust stats,
    # completeness estimates). `belief_map` is the receiver's CURRENT belief, so
    # a message is interpreted against everything fused before it this tick.
    tick: int
    belief_map: np.ndarray


@runtime_checkable
class ReceiverTrustPolicy(Protocol):
    def interpret(
        self, message: DeliveredMessage, context: ReceiveContext
    ) -> ReceiverAction: ...


class ReceiveInbox:
    """Per-robot queue of delivered-but-not-yet-interpreted messages."""

    def __init__(self) -> None:
        self._pending: list[DeliveredMessage] = []

    def enqueue(self, messages: Iterable[DeliveredMessage]) -> None:
        self._pending.extend(messages)

    def pending(self) -> tuple[DeliveredMessage, ...]:
        # Peek without consuming.
        return tuple(self._pending)

    def drain(self) -> list[DeliveredMessage]:
        # Return everything queued and clear the inbox.
        drained = self._pending
        self._pending = []
        return drained

    def __len__(self) -> int:
        return len(self._pending)


class TrustAllReceiver:
    """Baseline trust head: fuse every message at full trust.

    Reproduces the old env's "believe all teammates" behavior as the reference
    point a learned trust policy must improve on under degraded/adversarial comms.
    """

    def interpret(
        self, message: DeliveredMessage, context: ReceiveContext
    ) -> ReceiverAction:
        return ReceiverAction(ReceiverDecision.FUSE, trust=1.0)


def process_inbox(
    robot: "Robot",
    inbox: ReceiveInbox,
    policy: ReceiverTrustPolicy,
    *,
    tick: int,
) -> list[DeliveredMessage]:
    # Drain the inbox, interpret each message with the trust policy, and apply
    # the resulting decision. FUSE writes cells into the robot's belief with the
    # chosen trust; RELAY is collected and returned (recorded, not forwarded yet);
    # DISCARD is dropped. Messages are processed in delivery order, so each is
    # interpreted against belief as updated by the ones before it.
    relayed: list[DeliveredMessage] = []
    for message in inbox.drain():
        context = ReceiveContext(tick=tick, belief_map=robot.belief_map)
        action = policy.interpret(message, context)
        if action.decision is ReceiverDecision.FUSE:
            for position, value in message.cells:
                robot.fuse_cell(position, value, action.trust)
        elif action.decision is ReceiverDecision.RELAY:
            relayed.append(message)
        # DISCARD: nothing to do.
    return relayed
