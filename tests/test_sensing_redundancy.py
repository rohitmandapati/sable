"""Physical sensing redundancy: Environment.sensing_redundancy().

rho = (sum_r |cells r sensed| - |team-sensed union|) / |team-sensed union|,
counting each robot's own-sensor coverage (free + wall). It isolates
cross-robot duplicated sensing; 0 for a lone robot, bounded above by n-1.
"""

import numpy as np

from environment import Environment
from robot import UNKNOWN


def _run(num_robots, size=(12, 12), seed=0, density=0.2, max_ticks=2000):
    ids = [f"r{i}" for i in range(num_robots)]
    env = Environment(width=size[0], height=size[1], robot_ids=ids,
                      obstacle_density=density, max_ticks=max_ticks)
    obs, _ = env.reset(seed=seed)
    rngs = {rid: np.random.default_rng([seed, i]) for i, rid in enumerate(ids)}
    # Drive with the frontier baseline until coverage completes or ticks run out.
    from policy import make_policy
    policy = make_policy("move_toward_frontier_bfs")
    while env.agents:
        actions = {rid: policy.act(obs[rid], rngs[rid]) for rid in env.agents}
        obs, *_ = env.step(actions)
    return env


def test_single_robot_has_zero_redundancy():
    env = _run(1)
    assert env.sensing_redundancy() == 0.0


def test_redundancy_within_bounds_and_positive_for_team():
    n = 4
    env = _run(n)
    rho = env.sensing_redundancy()
    assert 0.0 <= rho <= n - 1
    assert rho > 0.0  # uncoordinated robots re-sense shared ground


def test_matches_belief_overlap_without_comms():
    # With no communication a robot's belief comes only from its own sensing, so
    # sensed_mask == (belief != UNKNOWN) and the physical metric equals the
    # belief-overlap it replaces. This identity is what the comms slice will
    # later break (belief inflates, sensing stays honest).
    env = _run(4)
    sum_known, union = 0, np.zeros((env.height, env.width), dtype=bool)
    for robot in env.robots.values():
        assert np.array_equal(robot.sensed_mask, robot.belief_map != UNKNOWN)
        known = robot.belief_map != UNKNOWN
        sum_known += int(known.sum())
        union |= known
    belief_overlap = (sum_known - int(union.sum())) / int(union.sum())
    assert env.sensing_redundancy() == belief_overlap


def test_disjoint_sensing_is_zero_overlapping_is_positive():
    # Direct control of the metric on a hand-built two-robot env.
    env = Environment(width=6, height=6, robot_ids=["a", "b"], obstacle_density=0.0)
    env.reset(seed=0)
    for r in env.robots.values():
        r.sensed_mask[:] = False

    # Disjoint sensed sets -> no cell double-sensed -> rho == 0.
    env.robots["a"].sensed_mask[1, 1] = True
    env.robots["b"].sensed_mask[3, 3] = True
    assert env.sensing_redundancy() == 0.0

    # One shared cell across the two robots -> rho > 0.
    env.robots["b"].sensed_mask[1, 1] = True
    assert env.sensing_redundancy() > 0.0
