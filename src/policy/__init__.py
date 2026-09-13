# Policy layer, the only place that knows concrete policy implementations

from typing import Callable

from actions import Action
from observations import RobotObservation
from policy.base import FunctionPolicy, Policy, PolicyFn
from policy.coordinated_frontier import CoordinatedFrontierPolicy
from policy.move_random_policy import move_random
from policy.move_toward_frontier import move_toward_frontier_bfs, move_toward_frontier_astar
from policy.move_toward_unknown_policy import (
    move_toward_unknown,
    move_toward_unknown_bfs,
)

POLICIES: dict[str, PolicyFn] = {
    "move_random": move_random,
    "move_toward_unknown": move_toward_unknown,
    "move_toward_unknown_bfs": move_toward_unknown_bfs,
    "move_toward_frontier_bfs": move_toward_frontier_bfs,
    "move_toward_frontier_astar": move_toward_frontier_astar,
}

# Coordinated (joint) policies take the whole team's observations at once, so
# they can't be plain PolicyFns. Each entry builds a fresh policy instance.
COORDINATED_POLICIES: dict[str, Callable[[], Policy]] = {
    "coordinated_frontier_greedy": lambda: CoordinatedFrontierPolicy(mode="greedy"),
    "coordinated_frontier_hungarian": lambda: CoordinatedFrontierPolicy(mode="hungarian"),
}


def make_policy(name: str) -> Policy:
    """Construct a Policy object (with `.act`/`.act_joint`) for a policy name."""
    if name in COORDINATED_POLICIES:
        return COORDINATED_POLICIES[name]()
    try:
        fn = POLICIES[name]
    except KeyError:
        available = sorted(POLICIES) + sorted(COORDINATED_POLICIES)
        raise KeyError(
            f"unknown policy {name!r}; available: {available}"
        ) from None
    return FunctionPolicy(fn, name=name)


__all__ = [
    "POLICIES",
    "COORDINATED_POLICIES",
    "make_policy",
    "Policy",
    "PolicyFn",
    "FunctionPolicy",
    "CoordinatedFrontierPolicy",
    "Action",
    "RobotObservation",
]
