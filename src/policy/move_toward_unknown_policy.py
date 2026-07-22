
import numpy as np
from src.actions import Action
from src.robot import KNOWN_WALL, UNKNOWN, Robot


def move_toward_unknown(robot: Robot, rng: np.random.Generator) -> Action:
        # Move toward unknown cells if possible
        if not robot.alive:
            raise RuntimeError("Inactive robot cannot move")
        row, col = robot.pos
        possible_actions = [
            (0, 0),   # Stay in place
            (-1, 0),  # Up
            (1, 0),   # Down
            (0, -1),  # Left
            (0, 1),   # Right
        ]
        valid_actions = [
            (dr, dc) for dr, dc in possible_actions
            if 0 <= row + dr < robot.map_shape[0]
            and 0 <= col + dc < robot.map_shape[1]
            and robot.belief_map[row + dr, col + dc] != KNOWN_WALL
        ]
        if not valid_actions:
            return (0, 0)  # No valid moves, stay in place

        # Prioritize actions that lead to unknown cells
        unknown_actions = [
            (dr, dc) for dr, dc in valid_actions
            if robot.belief_map[row + dr, col + dc] == UNKNOWN
        ]
        if unknown_actions:
            return unknown_actions[rng.choice(len(unknown_actions))]
        return valid_actions[rng.choice(len(valid_actions))]  # No unknown cells, pick randomly