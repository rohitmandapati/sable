# Phase-2 comms realism: distance path-loss + hard range cutoff + obstacle
# occlusion, all env-side physics evaluated per (sender, recipient) hop. Defaults
# leave the uniform model untouched (verified in test_link_model.py); here we pin
# the new physical causes and their per-cause attribution.

import numpy as np

from comms import (
    CAUSE_OCCLUDED,
    CAUSE_OUT_OF_RANGE,
    CAUSE_STOCHASTIC,
    CommsChannel,
    CommsConfig,
    LinkModel,
    LinkQuery,
)
from comms.link import _bresenham, _wall_count_between
from environment import Environment
from robot import KNOWN_FREE


def _q(a, b, size=6):
    return LinkQuery("s", "r", a, b, tick=0, size_bytes=size)


def _cells(n):
    return tuple(((0, c), KNOWN_FREE) for c in range(n))


# -- distance path-loss + hard range -------------------------------------------

def test_within_reliable_range_always_delivers():
    link = LinkModel(CommsConfig(comm_range=10.0, reliable_range=2.0), seed=0)
    # d = 1 (adjacent) <= reliable_range -> no distance loss, no uniform loss.
    assert all(link.evaluate(_q((0, 0), (0, 1))).delivered for _ in range(50))


def test_beyond_range_is_a_deterministic_out_of_range_drop():
    link = LinkModel(CommsConfig(comm_range=10.0), seed=0)
    out = link.evaluate(_q((0, 0), (0, 20)))  # d = 20 >= comm_range
    assert not out.delivered
    assert out.drop_cause == CAUSE_OUT_OF_RANGE


def test_midrange_loss_is_stochastic_and_grows_with_distance():
    # comm_range=10, reliable=2, falloff=2 -> p_dist = ((d-2)/8)^2.
    cfg = CommsConfig(comm_range=10.0, reliable_range=2.0, distance_falloff=2.0)

    def drop_fraction(d, n=600):
        link = LinkModel(cfg, seed=0)
        got = [link.evaluate(_q((0, 0), (0, d))).delivered for _ in range(n)]
        return 1.0 - sum(got) / n

    near = drop_fraction(4)   # p ~ (2/8)^2 = 0.0625
    far = drop_fraction(8)    # p ~ (6/8)^2 = 0.5625
    assert near < far
    assert 0.0 < near < 0.15
    assert 0.5 < far < 0.65


def test_no_positions_disables_distance_effects():
    link = LinkModel(CommsConfig(comm_range=1.0), seed=0)  # tiny range...
    # ...but with no positions the link can't measure distance, so it delivers.
    assert link.evaluate(_q(None, None)).delivered


# -- occlusion -----------------------------------------------------------------

def test_wall_count_between_excludes_endpoints():
    grid = np.zeros((1, 7), dtype=np.int8)
    grid[0, 3] = 1  # one wall mid-corridor
    assert _wall_count_between(grid, (0, 1), (0, 5)) == 1
    # Endpoints are never counted even if they sit on walls.
    assert _wall_count_between(grid, (0, 3), (0, 5)) == 0


def test_full_wall_attenuation_blocks_deterministically():
    grid = np.zeros((1, 7), dtype=np.int8)
    grid[0, 3] = 1
    link = LinkModel(CommsConfig(wall_attenuation=1.0), seed=0, grid=grid)
    out = link.evaluate(_q((0, 1), (0, 5)))
    assert not out.delivered
    assert out.drop_cause == CAUSE_OCCLUDED


def test_clear_line_of_sight_delivers():
    grid = np.zeros((1, 7), dtype=np.int8)  # no walls
    link = LinkModel(CommsConfig(wall_attenuation=1.0), seed=0, grid=grid)
    assert link.evaluate(_q((0, 1), (0, 5))).delivered


def test_partial_wall_attenuation_is_stochastic():
    grid = np.zeros((1, 7), dtype=np.int8)
    grid[0, 3] = 1
    link = LinkModel(CommsConfig(wall_attenuation=0.5), seed=0, grid=grid)
    got = [link.evaluate(_q((0, 1), (0, 5))).delivered for _ in range(600)]
    frac = 1.0 - sum(got) / len(got)
    assert 0.4 < frac < 0.6  # p_occ = 0.5


def test_bresenham_is_symmetric_endpoints():
    line = _bresenham((0, 0), (0, 4))
    assert line[0] == (0, 0) and line[-1] == (0, 4)
    assert len(line) == 5


# -- per-cause attribution via the channel -------------------------------------

def test_channel_attributes_stochastic_and_bandwidth_drops():
    ch = CommsChannel(LinkModel(CommsConfig(drop_prob=1.0), seed=0))
    ch.send("r0", _cells(1), recipients=["r1", "r2"], tick=0)
    assert ch.drops_by_cause[CAUSE_STOCHASTIC] == 2

    ch2 = CommsChannel(LinkModel(CommsConfig(max_bytes_per_tick=0)))
    ch2.send("r0", _cells(1), recipients=["r1"], tick=0)
    assert ch2.drops_by_cause["bandwidth"] == 1


# -- env integration: real positions drive the physics -------------------------

def test_tiny_range_drops_everything_out_of_range_in_env():
    # comm_range = 1.0: every distinct pair of robots is at Euclidean distance
    # >= 1.0, so the whole team is out of range -> nothing delivered, all drops
    # attributed to range. Confirms the env feeds real positions to the link.
    env = Environment(
        width=20, height=20, robot_ids=["a", "b", "c"], obstacle_density=0.0,
        enable_comms=True, comms_config=CommsConfig(comm_range=1.0),
    )
    env.reset(seed=3)
    from actions import Action
    for _ in range(5):
        env.step({rid: Action.STAY for rid in env.agents})
    assert env.comms.payload_bytes_delivered == 0
    assert env.comms.drops_by_cause[CAUSE_OUT_OF_RANGE] > 0
