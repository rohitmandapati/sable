# Policy layer, the only place that knows concrete policy implementations

from actions import Action
from observations import RobotObservation
from policy.base import FunctionPolicy, Policy, PolicyFn
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


def make_policy(name: str) -> Policy:
    """Construct a Policy object (with `.act`) for a registered policy name."""
    try:
        fn = POLICIES[name]
    except KeyError:
        raise KeyError(
            f"unknown policy {name!r}; available: {sorted(POLICIES)}"
        ) from None
    return FunctionPolicy(fn, name=name)


__all__ = [
    "POLICIES",
    "make_policy",
    "Policy",
    "PolicyFn",
    "FunctionPolicy",
    "Action",
    "RobotObservation",
]
