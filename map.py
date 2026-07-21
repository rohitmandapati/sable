import numpy as np

class Map:
    
    def __init__(self, width, height, seed_generator: np.uint8=None):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=int)
        self.rng = np.random.default_rng(seed=seed_generator)
    
    def generate_random_map(self, obstacle_density=0.2):
        pass