"""Unit tests for the policy-facing comms API (CommunicationAction / CommsBackend
/ DeliveredMessage). conftest.py puts src/ on sys.path, so imports are flat.

These pin the *contract* the learned comms policy relies on -- the same contract
every future (lossy/latency/adversarial) backend must honor.
"""

import pytest

from comms import (
    CommunicationAction,
    DeliveredMessage,
    PayloadKind,
    PerfectBroadcastBackend,
)
from comms.message import Message
from robot import KNOWN_FREE, KNOWN_WALL


def _cells(n):
    return tuple(((0, c), KNOWN_FREE) for c in range(n))


# -- CommunicationAction ---------------------------------------------------------

def test_skip_action_sends_nothing():
    backend = PerfectBroadcastBackend(["a", "b"])
    assert backend.execute(CommunicationAction.skip(), sender_id="a", tick=1) is None
    assert backend.deliver("b", tick=2) == []
    assert backend.stats.messages_transmitted == 0


def test_broadcast_is_not_addressed_to_self():
    assert CommunicationAction.broadcast(_cells(1)).is_broadcast
    assert not CommunicationAction.unicast(("b",), _cells(1)).is_broadcast


# -- PerfectBroadcastBackend: routing -------------------------------------------

def test_broadcast_reaches_every_peer_but_not_sender():
    backend = PerfectBroadcastBackend(["a", "b", "c"])
    backend.execute(CommunicationAction.broadcast(_cells(2)), sender_id="a", tick=1)
    assert backend.deliver("a", tick=2) == []          # never messages itself
    assert len(backend.deliver("b", tick=2)) == 1
    assert len(backend.deliver("c", tick=2)) == 1


def test_unicast_reaches_only_named_recipients():
    backend = PerfectBroadcastBackend(["a", "b", "c"])
    backend.execute(
        CommunicationAction.unicast(("b",), _cells(1)), sender_id="a", tick=1
    )
    assert len(backend.deliver("b", tick=2)) == 1
    assert backend.deliver("c", tick=2) == []


def test_send_to_endpoint_matches_unicast_action():
    backend = PerfectBroadcastBackend(["a", "b", "c"])
    backend.send_to("a", ["c"], _cells(1), tick=1)
    assert backend.deliver("b", tick=2) == []
    assert len(backend.deliver("c", tick=2)) == 1


# -- delivery payload + metadata ------------------------------------------------

def test_delivered_payload_round_trips_through_the_wire():
    cells = (((1, 2), KNOWN_FREE), ((3, 4), KNOWN_WALL))
    backend = PerfectBroadcastBackend(["a", "b"])
    backend.broadcast("a", cells, tick=1, sender_position=(7, 8))
    (msg,) = backend.deliver("b", tick=2)
    assert isinstance(msg, DeliveredMessage)
    assert msg.cells == cells                 # decoded exactly, via Message wire
    assert msg.sender_id == "a"
    assert msg.sender_position == (7, 8)       # carried as (unverified) metadata
    assert msg.payload_kind is PayloadKind.BELIEF_DELTA


def test_delivered_metadata_comes_off_the_wire_for_large_ticks():
    # Provenance is reconstructed from the wire frame, so ticks/ids beyond int16
    # survive intact.
    backend = PerfectBroadcastBackend(["a", "b"])
    backend.broadcast("a", _cells(1), tick=40_000)
    (msg,) = backend.deliver("b", tick=40_001)
    assert msg.sender_id == "a"
    assert msg.created_tick == 40_000


def test_age_reflects_created_vs_delivered_tick():
    backend = PerfectBroadcastBackend(["a", "b"])
    backend.broadcast("a", _cells(1), tick=3)
    (msg,) = backend.deliver("b", tick=5)
    assert msg.created_tick == 3
    assert msg.delivered_tick == 5
    assert msg.age == 2


def test_deliver_drains_the_inbox():
    backend = PerfectBroadcastBackend(["a", "b"])
    backend.broadcast("a", _cells(1), tick=1)
    assert len(backend.deliver("b", tick=2)) == 1
    assert backend.deliver("b", tick=3) == []   # nothing left after a drain


def test_sequence_ids_are_unique_and_monotonic():
    backend = PerfectBroadcastBackend(["a", "b"])
    backend.broadcast("a", _cells(1), tick=1)
    backend.broadcast("a", _cells(1), tick=1)
    first, second = backend.deliver("b", tick=2)
    assert second.sequence_id > first.sequence_id


# -- stats + reset --------------------------------------------------------------

def test_stats_account_payload_bytes_and_counts():
    backend = PerfectBroadcastBackend(["a", "b", "c"])
    backend.broadcast("a", _cells(1), tick=1)   # 1 cell = 6 payload bytes
    assert backend.stats.messages_transmitted == 1
    assert backend.stats.payload_bytes_transmitted == 6
    backend.deliver("b", tick=2)
    backend.deliver("c", tick=2)
    assert backend.stats.deliveries_made == 2   # one send, two recipients
    assert backend.stats.payload_bytes_delivered == 12


def test_reset_clears_inboxes_and_stats():
    backend = PerfectBroadcastBackend(["a", "b"])
    backend.broadcast("a", _cells(1), tick=1)
    backend.reset()
    assert backend.deliver("b", tick=2) == []
    assert backend.stats.messages_transmitted == 0
    assert backend.stats.payload_bytes_delivered == 0


def test_empty_payload_is_a_legal_send():
    backend = PerfectBroadcastBackend(["a", "b"])
    backend.broadcast("a", (), tick=1)
    (msg,) = backend.deliver("b", tick=2)
    assert msg.cells == ()
    assert msg.payload_size_bytes == 0
