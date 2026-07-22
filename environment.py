from collections.abc import Callable

from robot import Robot, Position
from map import Map


class Environment:
    def __init__(self, map: Map, robots: list[Robot]):
        self.map = map
        self.robots = robots
        
    
    def sample_spawns(self, num_samples: int):
        free_cells = self.map.free_cells
        if len(free_cells) < num_samples:
            raise ValueError("Not enough free cells to sample from.")
        
        sampled_indices = self.map.rng.choice(len(free_cells), size=num_samples, replace=False)
        print(free_cells[sampled_indices])
        # map.grid[tuple(free_cells[sampled_indices].T)] = 2  # Mark sampled spawns on the map, debug
        return free_cells[sampled_indices]
    
    def update_robot_positions(self, robot: Robot, new_position: Position) -> None:
        for robot in self.robots:
            if robot.alive:
                self.determine_next_move(robot, Robot.move_random())
                robot.set_position(new_position)
    
    
    def determine_next_move(self, robot: Robot, move_func: Callable[[], Position]) -> Position:
        if not robot.alive:
            raise RuntimeError("Inactive robot cannot determine next move")
        next_position = move_func()
        if not (0 <= next_position[0] < robot.map_shape[0]) or not (0 <= next_position[1] < robot.map_shape[1]):
            return robot.pos # out of bounds, ignore
        return next_position
        
    def tick() -> None:
        # Placeholder for tick logic
        pass
        
        