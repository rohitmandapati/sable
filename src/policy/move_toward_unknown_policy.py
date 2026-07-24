from __future__ import annotations

import numpy as np

from actions import Action
from observations import RobotObservation
from planning import is_frontier_cell
from policy.move_random_policy import move_random
from policy.move_toward_frontier import move_toward_frontier_bfs, move_toward_frontier_astar
from robot import KNOWN_WALL, UNKNOWN

_MOVES = (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)


# this policy is effectively equivalent to move_random
def move_toward_unknown(
    observation: RobotObservation, rng: np.random.Generator
) -> Action:
    """Prefer a step onto an unknown neighbor; otherwise a random valid step."""
    if not observation.alive:
        raise RuntimeError("Inactive robot cannot move")

    row, col = observation.position
    height, width = observation.map_shape
    belief = observation.belief_map

    valid = [
        move
        for move in (Action.STAY, *_MOVES)
        if 0 <= row + move.delta[0] < height
        and 0 <= col + move.delta[1] < width
        and belief[row + move.delta[0], col + move.delta[1]] != KNOWN_WALL
    ]
    if not valid:
        return move_random(observation, rng)

    unknown = [
        move
        for move in valid
        if belief[row + move.delta[0], col + move.delta[1]] == UNKNOWN
    ]
    if unknown:
        return unknown[rng.choice(len(unknown))]
    return valid[rng.choice(len(valid))]


# One step lookahead: if an adjacent traversable cell itself borders the
# unknown, step there immediately; otherwise defer to the BFS frontier planner.
def move_toward_unknown_bfs(
    observation: RobotObservation, rng: np.random.Generator
) -> Action:
    if not observation.alive:
        raise RuntimeError("Inactive robot cannot move")

    row, col = observation.position
    height, width = observation.map_shape
    belief = observation.belief_map

    for move in _MOVES:
        nr, nc = row + move.delta[0], col + move.delta[1]
        if not (0 <= nr < height and 0 <= nc < width):
            continue
        if belief[nr, nc] == KNOWN_WALL:
            continue
        if is_frontier_cell(belief, (nr, nc)):
            return move
    return move_toward_frontier_bfs(observation, rng)

# Same but with A*
def move_toward_unknown_astar(
    observation: RobotObservation, rng: np.random.Generator
) -> Action:
    if not observation.alive:
        raise RuntimeError("Inactive robot cannot move")

    row, col = observation.position
    height, width = observation.map_shape
    belief = observation.belief_map

    for move in _MOVES:
        nr, nc = row + move.delta[0], col + move.delta[1]
        if not (0 <= nr < height and 0 <= nc < width):
            continue
        if belief[nr, nc] == KNOWN_WALL:
            continue
        if is_frontier_cell(belief, (nr, nc)):
            return move
    return move_toward_frontier_astar(observation, rng)
