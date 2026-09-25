"""Unit + integration tests for the unified comms policy seam.

conftest.py puts src/ on sys.path, so imports are flat.

The env used to broadcast newly-sensed cells and fuse-everything inline. That
behavior now lives behind a swappable CommsPolicy (a send head + a receive/trust
head). These tests pin the three baselines and PROVE the baseline adapter
reproduces the original broadcast/fuse behavior bit-for-bit -- both directly and
when driven through the environment.
"""

import dataclasses

import numpy as np
import pytest

from actions import Action
from comms import (
    BroadcastNewCellsSender,
    CommsPolicy,
    CommunicationAction,
    CompositeCommsPolicy,
    NoCommsPolicy,
    PayloadKind,
    ReceiveContext,
    ReceiverDecision,
    SendContext,
    TrustAllReceiver,
    default_comms_policy,
)
from comms.delivered import DeliveredMessage
from environment import Environment
from robot import KNOWN_FREE, KNOWN_WALL, UNKNOWN, Robot


def _send_context(cells, *, rid="r0", tick=1, pos=(0, 0)):
    return SendContext(
        robot_id=rid,
        tick=tick,
        sensed_cells=tuple(cells),
        position=pos,
        belief_map=np.full((4, 4), UNKNOWN, dtype=np.int8),
    )


def _delivered(cells, *, sender="a", created=1, delivered=2, seq=0):
    return DeliveredMessage(
        sender_id=sender,
        payload_kind=PayloadKind.BELIEF_DELTA,
        cells=tuple(cells),
        created_tick=created,
        delivered_tick=delivered,
        sequence_id=seq,
        payload_size_bytes=len(cells) * 6,
    )


# -- BroadcastNewCellsSender ---------------------------------------------------

def test_broadcast_sender_broadcasts_new_cells():
    sender = BroadcastNewCellsSender()
    cells = (((1, 2), KNOWN_FREE), ((3, 3), KNOWN_WALL))
    action = sender.decide_send(_send_context(cells))
    assert action.send is True
    assert action.is_broadcast          # recipients=None => broadcast to all
    assert action.cells == cells
    assert action.payload_kind is PayloadKind.BELIEF_DELTA


def test_broadcast_sender_skips_when_nothing_new():
    sender = BroadcastNewCellsSender()
    action = sender.decide_send(_send_context(()))
    assert action.send is False         # skip: spends no bandwidth
    assert action.cells == ()


# -- NoCommsPolicy -------------------------------------------------------------

def test_no_comms_policy_never_sends_and_discards():
    policy = NoCommsPolicy()
    assert policy.decide_send(_send_context((((0, 0), KNOWN_FREE),))).send is False
    action = policy.interpret(
        _delivered([((0, 0), KNOWN_FREE)]),
        ReceiveContext(tick=2, belief_map=np.full((4, 4), UNKNOWN, dtype=np.int8)),
    )
    assert action.decision is ReceiverDecision.DISCARD


# -- CompositeCommsPolicy / default seam ---------------------------------------

def test_composite_delegates_to_each_head():
    policy = CompositeCommsPolicy(BroadcastNewCellsSender(), TrustAllReceiver())
    send = policy.decide_send(_send_context((((0, 0), KNOWN_FREE),)))
    assert send.is_broadcast
    recv = policy.interpret(
        _delivered([((0, 0), KNOWN_FREE)]),
        ReceiveContext(tick=2, belief_map=np.full((4, 4), UNKNOWN, dtype=np.int8)),
    )
    assert recv.decision is ReceiverDecision.FUSE
    assert recv.trust == 1.0


def test_default_policy_is_broadcast_plus_trust_all():
    policy = default_comms_policy()
    assert isinstance(policy, CompositeCommsPolicy)
    assert isinstance(policy.sender, BroadcastNewCellsSender)
    assert isinstance(policy.receiver, TrustAllReceiver)


def test_all_baselines_satisfy_the_comms_policy_protocol():
    # runtime_checkable: each baseline exposes both heads (decide_send + interpret).
    assert isinstance(default_comms_policy(), CommsPolicy)
    assert isinstance(NoCommsPolicy(), CommsPolicy)


# -- contexts are read-only / immutable ----------------------------------------

def test_send_context_belief_map_is_not_writeable():
    robot = Robot(robot_id="r", pos=(0, 0), map_shape=(4, 4))
    ctx = SendContext(
        robot_id="r",
        tick=1,
        sensed_cells=(),
        position=(0, 0),
        belief_map=robot.belief_map,
    )
    assert not ctx.belief_map.flags.writeable
    with pytest.raises(ValueError):
        ctx.belief_map[0, 0] = KNOWN_WALL


def test_send_context_cannot_mutate_the_live_robot_belief():
    # The policy sees the live belief but cannot corrupt it: writes raise, and the
    # robot's own array stays writeable and unchanged.
    robot = Robot(robot_id="r", pos=(0, 0), map_shape=(4, 4))
    robot.belief_map[1, 1] = KNOWN_FREE
    ctx = SendContext(
        robot_id="r", tick=1, sensed_cells=(), position=(0, 0), belief_map=robot.belief_map
    )
    with pytest.raises(ValueError):
        ctx.belief_map[2, 2] = KNOWN_WALL
    assert robot.belief_map[2, 2] == UNKNOWN          # untouched
    assert robot.belief_map.flags.writeable            # env can still write
    robot.belief_map[3, 3] = KNOWN_FREE                # ...proven
    assert robot.belief_map[3, 3] == KNOWN_FREE


def test_send_context_is_frozen():
    ctx = SendContext(
        robot_id="r",
        tick=1,
        sensed_cells=(),
        position=(0, 0),
        belief_map=np.full((4, 4), UNKNOWN, dtype=np.int8),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.tick = 5


def test_receive_context_belief_map_is_not_writeable():
    robot = Robot(robot_id="r", pos=(0, 0), map_shape=(4, 4))
    ctx = ReceiveContext(tick=2, belief_map=robot.belief_map)
    assert not ctx.belief_map.flags.writeable
    with pytest.raises(ValueError):
        ctx.belief_map[0, 0] = KNOWN_WALL
    assert robot.belief_map.flags.writeable  # source stays mutable for the env


def test_send_context_view_reflects_live_belief_reads():
    # Read-only must not mean stale: the view shares memory, so it reflects the
    # robot's current belief at read time (only WRITES through it are forbidden).
    robot = Robot(robot_id="r", pos=(0, 0), map_shape=(4, 4))
    ctx = SendContext(
        robot_id="r", tick=1, sensed_cells=(), position=(0, 0), belief_map=robot.belief_map
    )
    robot.belief_map[0, 0] = KNOWN_FREE
    assert ctx.belief_map[0, 0] == KNOWN_FREE


def test_env_survives_a_policy_that_tries_to_mutate_context_belief():
    # A misbehaving learned policy that tries to write through the context must fail
    # loudly rather than silently corrupting belief. Drive it through the real env.
    class _MutatingSender:
        def decide_send(self, context):
            context.belief_map[0, 0] = KNOWN_WALL  # illegal
            return CommunicationAction.skip()

        def interpret(self, message, context):
            return TrustAllReceiver().interpret(message, context)

    env = _env(enable_comms=True, comms_policy=_MutatingSender())
    with pytest.raises(ValueError):
        env.reset(seed=0)  # send happens during initial-sensing exchange


# -- integration: baseline adapter reproduces original env behavior -------------

_PATTERNS = {
    "r0": [Action.RIGHT, Action.DOWN, Action.RIGHT, Action.DOWN],
    "r1": [Action.LEFT, Action.UP, Action.LEFT, Action.UP],
}


def _run(env, ticks=8, seed=0):
    env.reset(seed=seed)
    for t in range(ticks):
        env.step({rid: pat[t % len(pat)] for rid, pat in _PATTERNS.items()})
    return env


def _env(**kw):
    return Environment(
        width=12, height=12, robot_ids=["r0", "r1"], obstacle_density=0.0, **kw
    )


def test_explicit_baseline_policy_matches_implicit_default_bit_for_bit():
    # Passing default_comms_policy() explicitly must be identical to enabling comms
    # with no policy argument (the env fills in the same baseline).
    implicit = _run(_env(enable_comms=True))
    explicit = _run(_env(enable_comms=True, comms_policy=default_comms_policy()))

    for rid in implicit.robots:
        a, b = implicit.robots[rid], explicit.robots[rid]
        assert np.array_equal(a.belief_map, b.belief_map)
        assert np.array_equal(a.trust_map, b.trust_map)
        assert np.array_equal(a.sensed_mask, b.sensed_mask)

    # ...and the comms accounting matches too.
    assert implicit.comms.stats.deliveries_made == explicit.comms.stats.deliveries_made
    assert (
        implicit.comms.stats.payload_bytes_delivered
        == explicit.comms.stats.payload_bytes_delivered
    )
    assert (
        implicit.comms.stats.wire_bytes_transmitted
        == explicit.comms.stats.wire_bytes_transmitted
    )


def test_baseline_policy_reproduces_broadcast_and_fuse():
    # The classical broadcast+trust-all baseline must still: fuse correct teammate
    # cells into belief with trust 1.0, never touch first-hand sensing.
    env = _run(_env(enable_comms=True, comms_policy=default_comms_policy()))
    found_received = False
    for robot in env.robots.values():
        received_only = (robot.belief_map != UNKNOWN) & ~robot.sensed_mask
        if received_only.any():
            found_received = True
            for r, c in np.argwhere(received_only):
                assert robot.belief_map[r, c] == env.map.grid[r, c]  # uncorrupted
                assert not robot.sensed_mask[r, c]                   # not first-hand
                assert robot.trust_map[r, c] == 1.0                  # trust-all
    assert found_received


def test_no_comms_policy_leaves_belief_equal_to_sensed():
    # Comms enabled (backend live) but driven by NoCommsPolicy: no traffic flows,
    # so belief must equal first-hand sensing and nothing is ever fused -- matching
    # a comms-off run even though the backend exists.
    env = _run(_env(enable_comms=True, comms_policy=NoCommsPolicy()))
    assert env.comms is not None
    for robot in env.robots.values():
        assert np.array_equal(robot.belief_map != UNKNOWN, robot.sensed_mask)
        assert not robot.trust_map.any()
    # Nothing was ever transmitted or delivered.
    assert env.comms.stats.messages_transmitted == 0
    assert env.comms.stats.deliveries_made == 0


def test_no_comms_policy_matches_comms_off_belief():
    # Stronger: NoCommsPolicy (backend live) yields the SAME per-robot belief as a
    # genuine comms-off run on the same seed + action script.
    off = _run(_env(enable_comms=False))
    none = _run(_env(enable_comms=True, comms_policy=NoCommsPolicy()))
    for rid in off.robots:
        assert np.array_equal(off.robots[rid].belief_map, none.robots[rid].belief_map)
        assert np.array_equal(off.robots[rid].sensed_mask, none.robots[rid].sensed_mask)
