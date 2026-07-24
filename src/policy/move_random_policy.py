from __future__ import annotations

import numpy as np

from actions import Action
from observations import RobotObservation
from robot import KNOWN_WALL

_MOVES = (Action.STAY, Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)


def move_random(observation: RobotObservation, rng: np.random.Generator) -> Action:
    if not observation.alive:
        raise RuntimeError("Inactive robot cannot move")

    row, col = observation.position
    height, width = observation.map_shape
    belief = observation.belief_map

    valid = [
        move
        for move in _MOVES
        if 0 <= row + move.delta[0] < height
        and 0 <= col + move.delta[1] < width
        and belief[row + move.delta[0], col + move.delta[1]] != KNOWN_WALL
    ]
    if not valid:
        return Action.STAY
    return valid[rng.choice(len(valid))]
