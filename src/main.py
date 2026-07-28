# Interactive demo: one robot exploring a rendered map under a chosen policy.
#
# Note the data flow: the environment produces observations, the policy turns an
# observation into an action, and only the environment mutates world state.

import numpy as np

from environment import Environment
from policy import make_policy
from renderer import Renderer

if __name__ == "__main__":
    SEED = 42

    env = Environment(
        width=20,
        height=15,
        robot_ids=["robert"],
        obstacle_density=0.2,
    )
    policy = make_policy("move_toward_frontier")
    policy_rng = np.random.default_rng(SEED)

    observations, _ = env.reset(seed=SEED)
    renderer = Renderer(env.map, cell_size=60, fps=10)

    terminated = env.coverage_complete()
    truncated = False
    while env.agents:
        actions = {
            rid: policy.act(observations[rid], policy_rng)
            for rid in env.active_robot_ids()
        }
        observations, rewards, terminated, truncated, info = env.step(actions)
        renderer.render(list(env.robots.values()))

    total_free = len(env.map.free_cells)
    print(f"Explored all {total_free} free cells in {env.tick_count} ticks")
    renderer.close()
