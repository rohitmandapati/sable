"""Environment-level seed handling: named-map selection, stream separation,
valid/unique spawns, and full-episode replay from a seed manifest.
"""

import numpy as np
import pytest

from default_maps import DEFAULT_MAPS
from environment import Environment
from policy import make_policy


def _spawns(env):
    return {rid: env.robots[rid].pos for rid in env.robots}


# -- named default maps ----------------------------------------------------


def test_named_map_via_constructor():
    env = Environment(width=10, height=10, robot_ids=["r0", "r1"], map_name="spiral")
    env.reset(seed=0)
    assert env.active_map_name == "spiral"
    assert np.array_equal(env.map.grid, np.array(DEFAULT_MAPS["spiral"]["grid"], dtype=int))


def test_named_map_via_options_overrides_default():
    env = Environment(width=10, height=10, robot_ids=["r0"], map_name="spiral")
    env.reset(seed=0, options={"map": "bordered_room"})
    assert env.active_map_name == "bordered_room"
    assert np.array_equal(env.map.grid, np.array(DEFAULT_MAPS["bordered_room"]["grid"], dtype=int))


def test_named_map_shape_mismatch_raises():
    env = Environment(width=20, height=15, robot_ids=["r0"], map_name="spiral")
    with pytest.raises(ValueError):
        env.reset(seed=0)


def test_named_map_reset_runs_through_environment():
    # A named-map episode should reset, sense, and step cleanly.
    env = Environment(width=10, height=10, robot_ids=["r0"], map_name="empty")
    obs, _ = env.reset(seed=1)
    assert env.observation_space("r0").contains(obs["r0"])
    policy = make_policy("move_toward_frontier_bfs")
    rng = np.random.default_rng(0)
    obs, *_ = env.step({"r0": policy.act(obs["r0"], rng)})
    assert env.observation_space("r0").contains(obs["r0"])


# -- stream separation -----------------------------------------------------


def test_same_root_reproduces_map_and_spawns():
    env_a = Environment(width=16, height=16, robot_ids=["r0", "r1", "r2"], obstacle_density=0.3)
    env_a.reset(seed=123)
    env_b = Environment(width=16, height=16, robot_ids=["r0", "r1", "r2"], obstacle_density=0.3)
    env_b.reset(seed=123)
    assert np.array_equal(env_a.map.grid, env_b.map.grid)
    assert _spawns(env_a) == _spawns(env_b)


def test_map_fixed_while_spawn_varies():
    # Overriding only the spawn stream keeps the identical map but moves robots.
    env = Environment(width=16, height=16, robot_ids=["r0", "r1", "r2"], obstacle_density=0.3)
    env.reset(seed=123)
    grid0, spawns0 = env.map.grid.copy(), _spawns(env)

    env.reset(seed=123, options={"seeds": {"spawn": 55}})
    assert np.array_equal(env.map.grid, grid0)      # same map
    assert _spawns(env) != spawns0                  # different spawns


def test_spawns_are_valid_and_unique():
    for seed in range(15):
        env = Environment(width=14, height=14, robot_ids=[f"r{i}" for i in range(5)],
                          obstacle_density=0.3)
        env.reset(seed=seed)
        positions = list(_spawns(env).values())
        assert len(set(positions)) == len(positions)  # unique
        for r, c in positions:
            assert env.map.grid[r, c] == 0            # on a free cell
            assert 0 <= r < env.height and 0 <= c < env.width


def test_manifest_records_streams_and_densities():
    env = Environment(width=12, height=12, robot_ids=["r0"], obstacle_density=0.3)
    env.reset(seed=None)
    manifest = env.episode_manifest()
    assert set(manifest["seeds"]) == {"root", "map", "spawn", "dynamics", "comms", "policy"}
    assert manifest["requested_obstacle_density"] == 0.3
    assert manifest["realized_obstacle_density"] != 0.3
    assert manifest["map_name"] is None


# -- full-episode replay ---------------------------------------------------


def _run_to_end(env, obs, max_steps=400):
    policy = make_policy("move_toward_frontier_bfs")
    rng = np.random.default_rng(0)
    steps = 0
    while env.agents and steps < max_steps:
        actions = {rid: policy.act(obs[rid], rng) for rid in env.agents}
        obs, *_ = env.step(actions)
        steps += 1
    return {rid: list(env.robots[rid].trajectory_map) for rid in env.robots}


def test_full_episode_replays_from_manifest():
    # Launch with seed=None, capture the manifest, then replay from its root.
    env1 = Environment(width=12, height=12, robot_ids=["r0", "r1"], obstacle_density=0.25)
    obs1, _ = env1.reset(seed=None)
    manifest = env1.episode_manifest()
    grid1, spawns1 = env1.map.grid.copy(), _spawns(env1)
    traj1 = _run_to_end(env1, obs1)

    env2 = Environment(width=12, height=12, robot_ids=["r0", "r1"], obstacle_density=0.25)
    obs2, _ = env2.reset(seed=manifest["seeds"]["root"])
    grid2, spawns2 = env2.map.grid.copy(), _spawns(env2)
    traj2 = _run_to_end(env2, obs2)

    assert np.array_equal(grid1, grid2)
    assert spawns1 == spawns2
    assert traj1 == traj2
