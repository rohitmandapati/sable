from dataclasses import *

import numpy as np
from map import Map

UNKNOWN = -1
KNOWN_FREE = 0
KNOWN_WALL = 1

Position = tuple[int, int]

@dataclass
class Robot:
    id:str
    pos: Position
    # map: Map # Must be VERY careful accessing this, don't want any accidental leakage but need to know the map the robot belongs to
    map_shape: tuple[int,int] = field(init=False)
    belief_map: np.ndarray = field(repr=False, init=False)
    trajectory_map: list[Position] = field(init=False)
    alive: bool = True
    
    
    
    def __post_init__(self) -> None:
        self.map_shape = (self.map.grid.shape[0], self.map.grid.shape[1])
        height, width = self.map_shape
        if height <= 0 or width <= 0:
            raise ValueError("Map dimensions must be positive")
        if not (0 <= self.pos[0] < height):
            raise ValueError("Robot row is outside the map")
        if not (0 <= self.pos[1] < width):
            raise ValueError("Robot column is outside the map")
        self.belief_map = np.full(self.map_shape, UNKNOWN, dtype=np.int8)
        self.trajectory_map = [self.pos]

    @property
    def position(self) -> Position:
        return self.pos

    def set_position(self, position: Position) -> None:
        if not self.alive:
            raise RuntimeError("Cannot move an inactive robot")
        row, col = position
        if not (0 <= row < self.map_shape[0]) or not (0 <= col < self.map_shape[1]):
            return
        self.pos=position
        self.trajectory_map.append(position)

    def reveal_cell(self, position: Position, value: int) -> None:
        if not self.alive:
            raise RuntimeError("Inactive robot cannot receive observations")
        
        row, col = position
        if not (0 <= row < self.map_shape[0]) or not (0 <= col < self.map_shape[1]):
            return # out of bounds, ignore
        if self.belief_map[row][col] == UNKNOWN:
            self.belief_map[row][col] = value
        else: return # already seen, ignore
            
    
    
    
    
    @staticmethod
    def sample_spawns(map: Map, num_samples: int, rng: np.random.Generator):
        free_cells = map.free_cells
        if len(free_cells) < num_samples:
            raise ValueError("Not enough free cells to sample from.")
        
        sampled_indices = rng.choice(len(free_cells), size=num_samples, replace=False)
        print(free_cells[sampled_indices])
        # map.grid[tuple(free_cells[sampled_indices].T)] = 2  # Mark sampled spawns on the map, debug
        return free_cells[sampled_indices]
    


    # def move_random(self):
    #     # Move the robot to a random adjacent cell (up, down, left, right) if it's free
    #     x, y = self.position
    #     possible_moves = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
    #     free_moves = [move for move in possible_moves if (move in self.map.free_cells)]
    #     if free_moves:
    #         self.position = free_moves[np.random.choice(len(free_moves))]

    # def mark_pos_observed(self):
    #     if self.position not in self.visited:
    #         self.visited.append(self.position)


