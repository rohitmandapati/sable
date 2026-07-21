from map import Map
from display import Display
from robot import Robot
import numpy as np

def tick():
    # Placeholder for tick logic
    pass


if __name__ == "__main__":
    map = Map(width=5, height=5, seed_generator=42, obstacle_density=0.3)
    rob = Robot(id=1,map=map, pos=(tuple(map.free_cells[np.random.choice(len(map.free_cells))])))

    display = Display(title="Sable", cell_size=60, fps=10)

