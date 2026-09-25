# One message as it lands in a recipient's inbox -- the input to the learned
# RECEIVE/TRUST head.
#
# Distinct from Message (the on-the-wire object): a DeliveredMessage adds the
# facts that only exist at *delivery* time -- when it arrived, and the transport
# metadata the trust head reasons over (claimed sender position, sequence id for
# duplicate detection, payload size for cost accounting). The trust head derives
# everything else (age, conflict-with-belief, staleness) from these plus the
# receiver's own belief; nothing here presumes the message is trustworthy.
#
# The payload `cells` are the DECODED belief patch (already round-tripped through
# the wire), so the receiver never touches raw bytes.

from __future__ import annotations

from dataclasses import dataclass

from comms.action import PayloadKind
from comms.message import Cell
from robot import Position


@dataclass(frozen=True)
class DeliveredMessage:
    sender_id: str
    payload_kind: PayloadKind
    cells: tuple[Cell, ...]      # decoded belief patch
    created_tick: int            # tick the sender emitted it
    delivered_tick: int          # tick it reached this inbox
    sequence_id: int             # sender-scoped message id (duplicate detection)
    payload_size_bytes: int      # payload-only size, for comms-cost accounting
    # Claimed sender position -- metadata, NOT verified. A spoofing backend can
    # later lie here; the trust head must treat it as a claim, not ground truth.
    sender_position: Position | None = None

    @property
    def age(self) -> int:
        # Ticks the message spent in flight. 0 only if delivered the same tick it
        # was created; the env drains inboxes at tick start, so live traffic is
        # age >= 1 (a message sent at t is consumable at t+1).
        return self.delivered_tick - self.created_tick
