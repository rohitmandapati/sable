from typing import List
import numpy as np

class Map:
    
    min_free_fraction = 0.3
    
    def __init__(self, width, height, seed_generator: np.uint8=None, obstacle_density=None):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=int)
        self.rng = np.random.default_rng(seed=seed_generator)
        self.vacancies = 0
        if obstacle_density is not None:
            self.generate_random_map(obstacle_density)
        else:
            self.generate_random_map()
        free_fraction = np.mean(self.grid == 0)
        while free_fraction < self.min_free_fraction:
            seed_generator += 1
            self.rng = np.random.default_rng(seed=seed_generator)
            self.generate_random_map(obstacle_density)
            free_fraction = np.mean(self.grid == 0)
        self.free_cells = np.argwhere(self.grid == 0)

            
    
    def generate_random_map(self, obstacle_density=0.5):
        # Generate a random map with obstacles represented as 1s and vacancies as 0s, flood fills all small disconnects
        if not (0 <= obstacle_density <= 1):
            raise ValueError("Obstacle density must be between 0 and 1.")
        self.grid = (self.rng.random((self.height, self.width)) < obstacle_density).astype(int)
        self.largest_connected_component()
        
        
        
    
    def largest_connected_component(self):
        visited = set()
        components = []
        for x in range(self.height):
            for y in range(self.width):
                if (x, y) in visited or self.grid[x, y] != 0:
                    continue
                connected, out, cells = self.flood_fill(x, y)
                components.append((connected, out))
                visited.update(cells) 
        if not components:
            return 0

        area, best = max(components, key=lambda c: c[0])
        self.grid = np.where(best == 2, 0, 1).astype(int)
        self.vacancies = area
        return area


    def flood_fill(self, x: int, y: int):
        if self.grid[x, y] == 1:
            return 0, np.zeros_like(self.grid), set()
        out = np.copy(self.grid)
        stack = [(x, y)]
        connected = 0
        cells = set()
        while stack:
            cx, cy = stack.pop()
            if cx < 0 or cx >= self.height or cy < 0 or cy >= self.width:
                continue
            if out[cx, cy] != 0:     
                continue
            out[cx, cy] = 2
            connected += 1
            cells.add((cx, cy))
            stack.append((cx + 1, cy))
            stack.append((cx - 1, cy))
            stack.append((cx, cy + 1))
            stack.append((cx, cy - 1))
        return connected, out, cells

    
    
    # Recursive flood fill implementation, outdated but kept for reference
    """
    def _flood_fill_helper_rec(self, x: int, y: int):
        if x < 0 or x >= self.height or y < 0 or y >= self.width:
            return
        if self.grid[x, y] != 0:
            return
        
        self.grid[x, y] = 2  # Mark the cell as filled
        
        # Recursively fill adjacent cells
        self._flood_fill_helper(x + 1, y)  # Right
        self._flood_fill_helper(x - 1, y)  # Left
        self._flood_fill_helper(x, y + 1)  # Up
        self._flood_fill_helper(x, y - 1)  # Down
    def flood_fill_rec(self, x: int, y: int):
        # 
        # Perform flood fill to find all connected spaces and set as 2
        # If (x,y) is an obstacle, raise exception
        # 
        if self.grid[x,y] == 1:
            raise ValueError("Starting point is an obstacle.")
        self._flood_fill_helper(x, y)"""