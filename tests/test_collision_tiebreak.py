"""Randomized tie-breaking in Environment._resolve_collisions.

An open (density 0.0) map has walled borders, so a 5x5 map is a connected 3x3
interior room (cells with row/col in 1..3) that we can place robots into by hand.
"""

from environment import Environment


def _open_env():
    env = Environment(width=5, height=5, robot_ids=["a", "b"], obstacle_density=0.0)
    env.reset(seed=0)
    return env


def test_standoff_is_broken_after_limit():
    env = _open_env()
    env.robots["a"].pos = (1, 1)
    env.robots["b"].pos = (1, 3)
    target = (1, 2)  # empty cell both robots race for
    desired = {"a": target, "b": target}

    # Below the limit: conservative rule holds both in place.
    assert env._resolve_collisions(desired) == {"a": (1, 1), "b": (1, 3)}
    assert env._resolve_collisions(desired) == {"a": (1, 1), "b": (1, 3)}

    # Past the limit (>2 consecutive contested ticks): exactly one proceeds.
    resolved = env._resolve_collisions(desired)
    winners = [rid for rid, pos in resolved.items() if pos == target]
    assert len(winners) == 1
    loser = next(rid for rid in resolved if rid not in winners)
    assert resolved[loser] == env.robots[loser].pos  # loser held in place


def test_cannot_enter_occupied_cell_even_after_limit():
    # If a robot sits on the target it holds it; nobody wins their way in, ever.
    env = _open_env()
    env.robots["a"].pos = (1, 1)
    env.robots["b"].pos = (1, 2)
    desired = {"a": (1, 2), "b": (1, 2)}  # a tries to move onto stationary b
    for _ in range(5):
        resolved = env._resolve_collisions(desired)
        assert resolved["a"] == (1, 1)
        assert resolved["b"] == (1, 2)


def test_contention_counter_resets_when_uncontested():
    env = _open_env()
    env.robots["a"].pos = (1, 1)
    env.robots["b"].pos = (1, 3)
    target = (1, 2)
    contested = {"a": target, "b": target}
    apart = {"a": (1, 1), "b": (1, 3)}  # no shared target

    env._resolve_collisions(contested)
    env._resolve_collisions(contested)
    env._resolve_collisions(apart)  # standoff broken; counter should decay
    # Two more contested ticks should NOT yet trigger a winner (counter restarted).
    assert env._resolve_collisions(contested) == {"a": (1, 1), "b": (1, 3)}
    assert env._resolve_collisions(contested) == {"a": (1, 1), "b": (1, 3)}
