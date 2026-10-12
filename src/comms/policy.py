# The comms POLICY seam -- one object that owns both learned comms heads.
#
# Communication is two decisions per robot per tick:
#   1. SEND  -- what (if anything) to transmit -> a CommunicationAction
#   2. RECEIVE/TRUST -- what to do with each delivered message -> a ReceiverAction
#
# The send side already speaks CommunicationAction (comms.action) and the receive
# side already speaks ReceiverAction (comms.receiver). This module unifies them
# into a single CommsPolicy the environment drives, so the whole comms behavior of
# a run is swappable as one object: classical baselines today, a learned MAPPO
# comms policy later (implementing the exact same two methods, reading the
# CommsObservation tensors). Nothing here is ML -- these are the classical
# reference heads the learned policy must beat.
#
# Layering: comms/ defines the seam and the baselines; learning/ will implement a
# learned CommsPolicy against it. The env depends only on this seam, never on a
# concrete policy.

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from comms.action import CommunicationAction
from comms.delivered import DeliveredMessage
from comms.message import Cell
from comms.receiver import (
    ReceiveContext,
    ReceiverAction,
    ReceiverDecision,
    ReceiverTrustPolicy,
    TrustAllReceiver,
    readonly_view,
)
from robot import Position


@dataclass(frozen=True)
class SendContext:
    # What the send head sees this tick. Small on purpose and mirrors
    # ReceiveContext on the receive side; the learned send head will grow this
    # (neighbor estimates, budget, message history) without changing the seam.
    # `sensed_cells` is this robot's newly-sensed belief delta -- the exact batch
    # the baseline broadcasts. The context is frozen and `belief_map` is exposed
    # read-only: deciding what to send must never mutate robot/world state, so a
    # future learned send head cannot corrupt the live belief through this view.
    # belief_map/position are provided for smarter senders; the baseline ignores them.
    robot_id: str
    tick: int
    sensed_cells: tuple[Cell, ...]
    position: Position
    belief_map: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "belief_map", readonly_view(self.belief_map))


@runtime_checkable
class SenderPolicy(Protocol):
    def decide_send(self, context: SendContext) -> CommunicationAction: ...


@runtime_checkable
class CommsPolicy(Protocol):
    # The full comms seam: a send head + a receive/trust head. `interpret` matches
    # ReceiverTrustPolicy exactly, so any CommsPolicy is also a valid receiver for
    # comms.process_inbox.
    def decide_send(self, context: SendContext) -> CommunicationAction: ...

    def interpret(
        self, message: DeliveredMessage, context: ReceiveContext
    ) -> ReceiverAction: ...


class BroadcastNewCellsSender:
    """Baseline send head: broadcast this tick's newly-sensed cells to everyone.

    Reproduces the environment's original inline send exactly: with nothing new to
    share it SKIPs (spends no bandwidth, transmits no message), otherwise it
    broadcasts the belief delta. The receiving side decides what to trust.
    """

    def decide_send(self, context: SendContext) -> CommunicationAction:
        if not context.sensed_cells:
            return CommunicationAction.skip()
        return CommunicationAction.broadcast(context.sensed_cells)


@dataclass
class CompositeCommsPolicy:
    """A CommsPolicy assembled from an independent send head and receive head.

    This is the adapter that lets the classical baselines -- BroadcastNewCellsSender
    on the send side, TrustAllReceiver on the receive side -- present as one policy
    the env can drive. Swap either head without touching the other.
    """

    sender: SenderPolicy
    receiver: ReceiverTrustPolicy

    def decide_send(self, context: SendContext) -> CommunicationAction:
        return self.sender.decide_send(context)

    def interpret(
        self, message: DeliveredMessage, context: ReceiveContext
    ) -> ReceiverAction:
        return self.receiver.interpret(message, context)


class NoCommsPolicy:
    """A CommsPolicy that communicates nothing: never sends, discards all receipts.

    Expresses "comms disabled" through the policy interface itself -- the backend
    may be live, but no traffic flows and no belief is ever fused, so behavior
    matches a run with comms off (belief == first-hand sensing).
    """

    def decide_send(self, context: SendContext) -> CommunicationAction:
        return CommunicationAction.skip()

    def interpret(
        self, message: DeliveredMessage, context: ReceiveContext
    ) -> ReceiverAction:
        return ReceiverAction(ReceiverDecision.DISCARD)


def default_comms_policy() -> CompositeCommsPolicy:
    # The env's default when comms is enabled but no policy is supplied: the exact
    # classical baseline the env used inline before this seam existed -- broadcast
    # newly-sensed cells, trust everything at full confidence.
    return CompositeCommsPolicy(BroadcastNewCellsSender(), TrustAllReceiver())
