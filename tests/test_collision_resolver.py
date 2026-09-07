"""Deterministic simultaneous-movement semantics for resolve_moves + env.step.

resolve_moves(current, desired) -> (final, conflicted) must always yield unique,
free final cells. The semantics under test (see resolve_moves docstring):

  * same-destination contention rejects all contenders;
  * two-robot swaps are rejected;
  * a convoy shifting into simultaneously-vacated cells succeeds iff it ends in
    an empty cell;
  * a chain ending at a stationary/blocked robot is rejected backward;
  * cycles (rotations) are rejected;
  * results are invariant to robot/dict iteration order;
  * random episodes keep every living robot on a unique, in-bounds, free cell.

There is no stochastic tie-breaking or backtracking any more: the environment
physics are fully deterministic.
"""

import itertools

import numpy as np

from actions import Action
from environment import Environment, resolve_moves


# -- unit tests on the pure resolver ---------------------------------------


def test_same_target_contention_rejects_all():
    current = {"a": (1, 1), "b": (1, 3)}
    desired = {"a": (1, 2), "b": (1, 2)}
    final, conflicted = resolve_moves(current, desired)
    assert final == current  # both held
    assert conflicted == {"a", "b"}


def test_two_robot_swap_is_rejected():
    current = {"a": (1, 1), "b": (1, 2)}
    desired = {"a": (1, 2), "b": (1, 1)}
    final, conflicted = resolve_moves(current, desired)
    assert final == current
    assert conflicted == {"a", "b"}


def test_legal_convoy_into_empty_cell():
    # a->b->c->empty: the whole convoy shifts one cell.
    current = {"a": (1, 1), "b": (1, 2), "c": (1, 3)}
    desired = {"a": (1, 2), "b": (1, 3), "c": (1, 4)}
    final, conflicted = resolve_moves(current, desired)
    assert final == {"a": (1, 2), "b": (1, 3), "c": (1, 4)}
    assert conflicted == set()


def test_chain_ending_at_stationary_robot_rejected_backward():
    # c requests its own cell (stays); b cannot enter it, so a cannot enter b's.
    current = {"a": (1, 1), "b": (1, 2), "c": (1, 3)}
    desired = {"a": (1, 2), "b": (1, 3), "c": (1, 3)}
    final, conflicted = resolve_moves(current, desired)
    assert final == current  # everyone held
    assert conflicted == {"a", "b"}  # c never wanted to move


def test_prompt_regression_no_overlap():
    # The case the old resolver got wrong: it could leave a and b both at (2,2).
    current = {"a": (2, 1), "b": (2, 2), "c": (2, 3)}
    desired = {"a": (2, 2), "b": (2, 3), "c": (2, 3)}
    final, conflicted = resolve_moves(current, desired)
    assert final == current
    assert len(set(final.values())) == len(final)  # unique -> no overlap


def test_chain_ending_at_wall_rejected_backward():
    # The caller validates a wall move into "stay", so c arrives with
    # desired == current; the convoy behind it must not move either.
    current = {"a": (1, 1), "b": (1, 2), "c": (1, 3)}
    desired = {"a": (1, 2), "b": (1, 3), "c": (1, 3)}  # c blocked by a wall
    final, _ = resolve_moves(current, desired)
    assert final == current


def test_three_body_dependency_partial_progress():
    # d moves into an empty cell; the convoy c->d's-cell->... behind it follows,
    # but an independent same-target pair (x, y) is rejected on the same tick.
    current = {"c": (0, 0), "d": (0, 1), "x": (5, 5), "y": (5, 7)}
    desired = {"c": (0, 1), "d": (0, 2), "x": (5, 6), "y": (5, 6)}
    final, conflicted = resolve_moves(current, desired)
    assert final["c"] == (0, 1) and final["d"] == (0, 2)  # convoy advances
    assert final["x"] == (5, 5) and final["y"] == (5, 7)  # contenders held
    assert conflicted == {"x", "y"}


def test_three_cycle_rotation_is_rejected():
    # A rotation reaches no empty cell, so -- like a swap -- it is rejected.
    current = {"a": (0, 0), "b": (0, 1), "c": (1, 1)}
    desired = {"a": (0, 1), "b": (1, 1), "c": (0, 0)}
    final, conflicted = resolve_moves(current, desired)
    assert final == current
    assert conflicted == {"a", "b", "c"}


def test_result_is_order_invariant():
    # The convoy-plus-contenders scenario must resolve identically regardless of
    # the order robots appear in the input dicts.
    base_current = {"a": (2, 1), "b": (2, 2), "c": (2, 3), "x": (5, 5), "y": (5, 7)}
    base_desired = {"a": (2, 2), "b": (2, 3), "c": (2, 4), "x": (5, 6), "y": (5, 6)}
    reference = resolve_moves(base_current, base_desired)[0]
    for order in itertools.permutations(base_current):
        current = {rid: base_current[rid] for rid in order}
        desired = {rid: base_desired[rid] for rid in order}
        final, _ = resolve_moves(current, desired)
        assert final == reference


def test_final_positions_always_unique_and_from_valid_set():
    # A dense scramble: everything either advances or holds, never overlaps.
    current = {"a": (1, 1), "b": (1, 2), "c": (1, 3), "d": (1, 4)}
    desired = {"a": (1, 2), "b": (1, 3), "c": (1, 4), "d": (1, 5)}  # (1,5) empty
    final, conflicted = resolve_moves(current, desired)
    assert len(set(final.values())) == len(final)
    assert conflicted == set()


# -- integration: random episodes must never overlap -----------------------


def _all_free_cells(env):
    return {(int(r), int(c)) for r, c in env.map.free_cells}


def test_random_episodes_keep_positions_unique_free_inbounds():
    free = None
    for seed in range(12):
        env = Environment(
            width=8, height=8, robot_ids=[f"r{i}" for i in range(4)],
            obstacle_density=0.2, max_ticks=200,
        )
        env.reset(seed=seed)
        free = _all_free_cells(env)
        rng = np.random.default_rng(seed)
        actions_list = list(Action.action_space())
        for _ in range(120):
            if not env.agents:
                break
            actions = {rid: actions_list[rng.integers(len(actions_list))] for rid in env.agents}
            env.step(actions)
            positions = [env.robots[rid].pos for rid in env.active_robot_ids()]
            # unique
            assert len(set(positions)) == len(positions)
            # in-bounds and free (never on a wall)
            for r, c in positions:
                assert 0 <= r < env.height and 0 <= c < env.width
                assert (r, c) in free


# -- integration: outcome counters ----------------------------------------


def test_wall_bump_is_counted_and_no_move():
    # A lone robot driven straight into the top wall stays and is counted.
    env = Environment(width=5, height=5, robot_ids=["a"], obstacle_density=0.0)
    env.reset(seed=0)
    env.robots["a"].pos = (1, 1)
    # Repeatedly try to go UP through the border wall at row 0.
    for _ in range(3):
        env.robots["a"].pos = (1, 1)
        _, _, _, _, infos = env.step({"a": Action.UP})
        assert env.robots["a"].pos == (1, 1)
        assert infos["a"]["wall_blocked"] is True
    assert env.wall_bumps == 3


def test_conflict_is_counted_and_reported():
    env = Environment(width=5, height=5, robot_ids=["a", "b"], obstacle_density=0.0)
    env.reset(seed=0)
    env.robots["a"].pos = (1, 1)
    env.robots["b"].pos = (1, 3)
    # Both race the empty cell (1,2): same-target contention holds both.
    _, _, _, _, infos = env.step({"a": Action.RIGHT, "b": Action.LEFT})
    assert env.robots["a"].pos == (1, 1)
    assert env.robots["b"].pos == (1, 3)
    assert infos["a"]["blocked"] is True and infos["b"]["blocked"] is True
    assert env.conflicts == 2
