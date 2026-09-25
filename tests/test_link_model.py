# Phase-1 comms plumbing: the CommsConfig + LinkModel.evaluate(query) seam that
# later realism stages (distance, occlusion, noise, latency, partial loss) extend.
# The uniform link's behaviour must be unchanged; these tests pin the new API and
# the fact that positions/grid now flow through to the link.

import numpy as np
import pytest

from comms import CommsChannel, CommsConfig, LinkModel, LinkOutcome, LinkQuery
from robot import KNOWN_FREE


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
    )
    base.update(kw)
    return LinkQuery(**base)


# -- CommsConfig ----------------------------------------------------------------

def test_config_defaults_describe_a_perfect_link():
    c = CommsConfig()
    assert c.drop_prob == 0.0
    assert c.max_bytes_per_tick is None


def test_config_rejects_out_of_range_values():
    with pytest.raises(ValueError):
        CommsConfig(drop_prob=1.5)
    with pytest.raises(ValueError):
        CommsConfig(drop_prob=-0.1)
    with pytest.raises(ValueError):
        CommsConfig(max_bytes_per_tick=-1)


# -- LinkModel.evaluate ---------------------------------------------------------

def test_lossless_link_always_delivers():
    link = LinkModel(CommsConfig(), seed=0)
    assert all(link.evaluate(_query()).delivered for _ in range(10))


def test_total_loss_never_delivers():
    link = LinkModel(CommsConfig(drop_prob=1.0), seed=0)
    assert not any(link.evaluate(_query()).delivered for _ in range(10))


def test_outcome_defaults_to_same_tick_delivery():
    assert LinkOutcome(delivered=True).delay_ticks == 0


def test_partial_loss_is_reproducible_by_seed():
    def deliveries(seed):
        link = LinkModel(CommsConfig(drop_prob=0.5), seed=seed)
        return [link.evaluate(_query()).delivered for _ in range(30)]

    same = deliveries(42)
    assert deliveries(42) == same          # deterministic per seed
    assert deliveries(7) != same           # different seed -> different stream
    assert any(same) and not all(same)     # a real mix of drops and deliveries


def test_default_max_bytes_reads_from_config():
    assert LinkModel(CommsConfig(max_bytes_per_tick=12)).max_bytes_per_tick == 12
    assert LinkModel(CommsConfig()).max_bytes_per_tick is None


def test_link_stores_grid_for_later_stages():
    grid = np.zeros((4, 4), dtype=np.int8)
    link = LinkModel(CommsConfig(), grid=grid)
    assert link.grid is grid


# -- channel -> link plumbing ---------------------------------------------------

class _SpyLink(LinkModel):
    def __init__(self):
        super().__init__(CommsConfig())
        self.queries: list[LinkQuery] = []

    def evaluate(self, query):
        self.queries.append(query)
        return super().evaluate(query)


def test_channel_threads_positions_and_size_into_query():
    link = _SpyLink()
    ch = CommsChannel(link)
    positions = {"r0": (1, 1), "r1": (2, 5)}
    ch.send("r0", _cells(1), recipients=["r1"], tick=3, positions=positions)

    assert len(link.queries) == 1
    q = link.queries[0]
    assert (q.sender_id, q.recipient_id) == ("r0", "r1")
    assert q.sender_pos == (1, 1)
    assert q.recipient_pos == (2, 5)
    assert q.tick == 3
    assert q.size_bytes == 6  # one cell


def test_channel_send_without_positions_still_works():
    # Positions are optional (unit callers omit them); the uniform link ignores
    # them anyway.
    link = _SpyLink()
    ch = CommsChannel(link)
    ch.send("r0", _cells(1), recipients=["r1"], tick=0)
    q = link.queries[0]
    assert q.sender_pos is None and q.recipient_pos is None
