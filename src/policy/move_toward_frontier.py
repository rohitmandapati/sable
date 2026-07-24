from __future__ import annotations

import numpy as np

from actions import Action
from observations import RobotObservation
from planning import find_frontier_cells, first_step_action, shortest_path_to_any, shortest_path_to_any_astar


def move_toward_frontier_bfs(observation: RobotObservation, rng: np.random.Generator) -> Action:
    if not observation.alive:
        raise RuntimeError("Inactive robot cannot move")

    frontier = find_frontier_cells(observation.belief_map)
    if not frontier:
        return Action.STAY

    path = shortest_path_to_any(observation.position, frontier, observation.belief_map)
    if path is None or len(path) < 2:
        return Action.STAY

    return first_step_action(path[0], path[1])


def move_toward_frontier_astar(observation: RobotObservation, rng: np.random.Generator) -> Action:
    if not observation.alive:
        raise RuntimeError("Inactive robot cannot move")

    frontier = find_frontier_cells(observation.belief_map)
    if not frontier:
        return Action.STAY

    path = shortest_path_to_any_astar(observation.position, frontier, observation.belief_map)
    if path is None or len(path) < 2:
        return Action.STAY

    return first_step_action(path[0], path[1])