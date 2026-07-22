from map import Map
from robot import Robot, KNOWN_FREE
from environment import Environment
from renderer import Renderer

if __name__ == "__main__":
    world = Map(width=20, height=15, seed=42, obstacle_density=0.2)
    start = tuple(world.free_cells[world.rng.choice(len(world.free_cells))])
    rob = Robot(robot_id="robert", pos=start, map_shape=(world.height, world.width))

    env = Environment(map=world, robots=[rob])
    renderer = Renderer(world, cell_size=60, fps=10)

    env.observe(rob) # sense at spawn so the first move respects walls
    total_free = len(world.free_cells)
    explored = lambda: all(rob.belief_map[r, c] == KNOWN_FREE for r, c in world.free_cells)

    while not explored():
        env.update_robot_positions(policy="move_toward_unknown_bfs")
        env.tick_count += 1
        renderer.render([rob])

    print(f"Explored all {total_free} free cells in {env.tick_count} ticks")
    renderer.close()
