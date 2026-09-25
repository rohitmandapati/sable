# Stage-3 comms realism: semi-random latency. Delivered messages arrive
# `delay_ticks` after they are sent; the delay is a physics-shaped mean plus an
# exponential jitter drawn from a per-message RNG keyed on the message id. These
# tests pin (a) the delay model, (b) its determinism/order-independence, (c) that
# it never perturbs the drop stream, and (d) that the channel honours the schedule.

import numpy as np
import pytest

from comms import CommsChannel, CommsConfig, LinkModel
from comms.link import LinkQuery
from robot import KNOWN_FREE, KNOWN_WALL


def _cells(n):
    return tuple(((0, c), KNOWN_FREE) for c in range(n))


def _query(size=6, **kw):
    base = dict(
        sender_id="r0",
        recipient_id="r1",
        sender_pos=None,
        recipient_pos=None,
        tick=0,
        size_bytes=size,
        message_id=0,
    )
    base.update(kw)
    return LinkQuery(**base)


# -- config validation ----------------------------------------------------------

def test_config_rejects_negative_latency_knobs():
    for kw in (
        {"latency_base": -1.0},
        {"latency_per_distance": -0.1},
        {"latency_per_wall": -0.1},
        {"latency_jitter": -0.5},
        {"max_latency_ticks": -1},
    ):
        with pytest.raises(ValueError):
            CommsConfig(**kw)


def test_default_link_delivers_same_tick():
    link = LinkModel(CommsConfig(), seed=0)
    assert link.evaluate(_query()).delay_ticks == 0


# -- the delay model ------------------------------------------------------------

def test_constant_base_delay_needs_no_rng():
    link = LinkModel(CommsConfig(latency_base=3.0))  # no seed, no jitter
    assert link.evaluate(_query()).delay_ticks == 3


def test_distance_term_scales_delay():
    link = LinkModel(CommsConfig(latency_per_distance=2.0))
    q = _query(sender_pos=(0, 0), recipient_pos=(0, 4))  # dist 4 -> 8 ticks
    assert link.evaluate(q).delay_ticks == 8


def test_wall_term_scales_delay():
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[0, 2] = KNOWN_WALL  # one wall strictly between the endpoints
    link = LinkModel(CommsConfig(latency_per_wall=5.0), grid=grid)
    q = _query(sender_pos=(0, 0), recipient_pos=(0, 4))
    assert link.evaluate(q).delay_ticks == 5


def test_max_latency_clamps():
    link = LinkModel(CommsConfig(latency_base=100.0, max_latency_ticks=10))
    assert link.evaluate(_query()).delay_ticks == 10


def test_jitter_produces_a_spread_of_delays():
    link = LinkModel(CommsConfig(latency_base=5.0, latency_jitter=5.0), seed=1)
    delays = [link.evaluate(_query(message_id=i)).delay_ticks for i in range(50)]
    assert all(d >= 0 for d in delays)
    assert len(set(delays)) > 1  # jitter actually varies the delay


# -- determinism: keyed on the message id, independent of call order ------------

def test_delay_is_reproducible_by_seed_and_message_id():
    def delay(seed, mid):
        link = LinkModel(CommsConfig(latency_jitter=5.0), seed=seed)
        return link.evaluate(_query(message_id=mid)).delay_ticks

    assert delay(3, 7) == delay(3, 7)      # same (seed, id) -> same delay
    # A different seed almost surely shifts at least one draw.
    assert any(delay(3, m) != delay(9, m) for m in range(20))


def test_delay_is_independent_of_evaluation_order():
    cfg = CommsConfig(latency_jitter=5.0)
    solo = LinkModel(cfg, seed=1).evaluate(_query(message_id=99)).delay_ticks

    busy = LinkModel(cfg, seed=1)
    for i in range(50):  # draw a pile of other messages first
        busy.evaluate(_query(message_id=i))
    after = busy.evaluate(_query(message_id=99)).delay_ticks
    assert solo == after  # message id, not call order, determines the delay


def test_latency_does_not_perturb_the_drop_stream():
    def deliveries(**latency):
        link = LinkModel(CommsConfig(drop_prob=0.5, **latency), seed=42)
        return [link.evaluate(_query(message_id=i)).delivered for i in range(40)]

    # Turning latency on must leave the drop decisions byte-for-byte identical.
    assert deliveries() == deliveries(latency_base=3.0, latency_jitter=5.0)


def test_reset_reproduces_the_same_jitter():
    link = LinkModel(CommsConfig(latency_jitter=5.0), seed=5)
    first = [link.evaluate(_query(message_id=i)).delay_ticks for i in range(20)]
    link.reset()
    again = [link.evaluate(_query(message_id=i)).delay_ticks for i in range(20)]
    assert first == again


# -- channel honours the schedule ----------------------------------------------

def test_channel_delivers_only_on_the_arrival_tick():
    ch = CommsChannel(LinkModel(CommsConfig(latency_base=2.0)))
    ch.send("r0", _cells(1), recipients=["r1"], tick=0)  # deliver_at = 2
    assert ch.messages_in_flight() == 1
    assert ch.receive("r1", tick=0) == []   # too early
    assert ch.receive("r1", tick=1) == []   # still latent
    got = ch.receive("r1", tick=2)          # arrives
    assert len(got) == 1
    assert ch.messages_in_flight() == 0
    assert ch.deliveries_made == 1
    assert ch.mean_delay == 2.0
    assert ch.delay_max == 2


def test_no_latency_delivers_same_tick_through_channel():
    ch = CommsChannel(LinkModel(CommsConfig()))
    ch.send("r0", _cells(1), recipients=["r1"], tick=0)
    assert len(ch.receive("r1", tick=0)) == 1  # delay 0 -> same tick
    assert ch.mean_delay == 0.0
