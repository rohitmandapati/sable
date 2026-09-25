# SABLE communication layer.
#
# One seam: the CommsBackend (the policy-facing transport API) with
# CommunicationAction / DeliveredMessage on the send side and the receive/trust
# side (ReceiveInbox -> ReceiverTrustPolicy -> fusion). The learned comms policy
# talks only to this seam; realism (loss, latency, spoofing) lives entirely in
# swappable backend implementations. PerfectBroadcastBackend is the lossless stub.

from comms.action import CommunicationAction, PayloadKind
from comms.backend import CommsBackend, CommsStats, PerfectBroadcastBackend
from comms.delivered import DeliveredMessage
from comms.message import Cell, Message
from comms.receiver import (
    ReceiveContext,
    ReceiveInbox,
    ReceiverAction,
    ReceiverDecision,
    ReceiverTrustPolicy,
    TrustAllReceiver,
    process_inbox,
)

__all__ = [
    # wire message
    "Cell",
    "Message",
    # policy-facing comms API (send side)
    "CommsBackend",
    "CommsStats",
    "PerfectBroadcastBackend",
    "CommunicationAction",
    "PayloadKind",
    "DeliveredMessage",
    # receive / trust side
    "ReceiveInbox",
    "ReceiveContext",
    "ReceiverAction",
    "ReceiverDecision",
    "ReceiverTrustPolicy",
    "TrustAllReceiver",
    "process_inbox",
]
