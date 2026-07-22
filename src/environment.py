from collections.abc import Callable

from robot import Robot, Position
from map import Map
import numpy as np
from actions import Action

from policy import POLICIES

class Environment:
    def __init__(self, map: Map, robots: list[Robot]):
        self.map = map
        self.robots = robots
        self.tick_count = 0
        
    
    def sample_spawns(self, num_samples: int):
        free_cells = self.map.free_cells
        if len(free_cells) < num_samples:
            raise ValueError("Not enough free cells to sample from.")
        
        sampled_indices = self.map.rng.choice(len(free_cells), size=num_samples, replace=False)
        print(free_cells[sampled_indices])
        # map.grid[tuple(free_cells[sampled_indices].T)] = 2  # Mark sampled spawns on the map, debug
        return free_cells[sampled_indices]
    
    def update_robot_positions(self, policy: str = "move_random", *args, **kwargs) -> None:
        for robot in self.robots:
            if not robot.alive:
                continue
            move_func = POLICIES[policy]
            self.observe(robot)

            robot.set_position(self.determine_next_move(robot, move_func, *args, **kwargs))


    def observe(self, robot: Robot) -> None:
        # Reveal the ground-truth value of the robot's cell and its orthogonal neighbors
        row, col = robot.position
        for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            r, c = row + dr, col + dc
            if 0 <= r < self.map.height and 0 <= c < self.map.width:
                robot.reveal_cell((r, c), int(self.map.grid[r, c]))

    def determine_next_move(
        self, robot: Robot, move_func: Callable[..., Action], *args, **kwargs
    ) -> Position:
        if not robot.alive:
            raise RuntimeError("Inactive robot cannot determine next move")

        dr, dc = move_func(robot, np.random.Generator(np.random.PCG64()), *args, **kwargs)
        r, c = robot.position
        next_position = (r + dr, c + dc)

        if not (0 <= next_position[0] < robot.map_shape[0]) or not (0 <= next_position[1] < robot.map_shape[1]):
            return robot.position # out of bounds
        return next_position

        
    def tick(self) -> None:
        self.update_robot_positions(policy="move_random")
        self.tick_count += 1

