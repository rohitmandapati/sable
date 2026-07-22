from map import Map
from display import Display
from robot import Robot
import numpy as np

def tick():
    # Placeholder for tick logic
    pass


if __name__ == "__main__":
    map = Map(width=10, height=10, seed=42, obstacle_density=0.2)
    rob = Robot(robot_id="robert", pos=(tuple(map.free_cells[np.random.choice(len(map.free_cells))])), map_shape=(map.height, map.width))

    display = Display(title="Sable", cell_size=60, fps=10)

    
