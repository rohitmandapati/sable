import numpy as np

class Map:
    
    def __init__(self, width, height, seed_generator: np.uint8=None, obstacle_density=None):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=int)
        self.rng = np.random.default_rng(seed=seed_generator)
        self.generate_random_map(obstacle_density)
    
    def generate_random_map(self, obstacle_density=0.5):
        """
        Generate a random map with obstacles represented as 1s and vacancies as 0s
        """
        if not (0 <= obstacle_density <= 1):
            raise ValueError("Obstacle density must be between 0 and 1.")
        
        self.grid = (self.rng.random((self.height, self.width)) < obstacle_density).astype(int)        
        # Generate random values and set cells to 1 (obstacle) based on the density
    
    def flood_fill(self, x: int, y: int):
        """
        Perform flood fill to find all connected spaces and set as 2
        If (x,y) is an obstacle, raise exception
        """
        if self.grid[x,y] == 1:
            raise ValueError("Starting point is an obstacle.")
        self._flood_fill_helper(x, y)
        
    def _flood_fill_helper(self, x: int, y: int):
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