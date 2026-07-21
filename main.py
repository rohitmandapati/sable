from map import Map
from display import Display

if __name__ == "__main__":
    map = Map(width=10, height=10, seed_generator=4, obstacle_density=0.3)    
    map.flood_fill(0, 0)
    Display().show(map)