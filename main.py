from map import Map
from display import Display

if __name__ == "__main__":
    map = Map(width=30, height=30, seed_generator=42, obstacle_density=0.3)    
    map.largest_connected_component()
    Display().show(map)