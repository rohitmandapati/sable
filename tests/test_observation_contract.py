"""PettingZoo observation-contract conformance for Environment.

The env declares a Gym `Dict` observation space but must actually *return*
observations that satisfy it, that are true snapshots (no memory shared with the
live robot belief, and immune to later steps), and that policies can consume
through the observation boundary alone -- no Robot, Map, or Environment handle.
"""

import numpy as np

from actions import Action
from environment import Environment
from observations import RobotObservation
from policy import make_policy
from robot import KNOWN_FREE, UNKNOWN


def _env(**kwargs):
    params = dict(width=10, height=8, robot_ids=["r0", "r1"], obstacle_density=0.2)
    params.update(kwargs)
    env = Environment(**params)
    obs, _ = env.reset(seed=7)
    return env, obs


def test_reset_observations_match_space():
    env, obs = _env()
    for rid in env.agents:
        assert env.observation_space(rid).contains(obs[rid])


def test_step_observations_match_space():
    env, obs = _env()
    rng = np.random.default_rng(0)
    policy = make_policy("move_toward_frontier_bfs")
    for _ in range(15):
        actions = {rid: policy.act(obs[rid], rng) for rid in env.agents}
        obs, *_ = env.step(actions)
        for rid in env.agents:
            assert env.observation_space(rid).contains(obs[rid])


def test_observation_dtype_and_shape():
    env, obs = _env()
    o = obs["r0"]
    assert o["belief_map"].dtype == np.int8
    assert o["belief_map"].shape == (env.height, env.width)
    assert o["position"].dtype == np.int64
    assert o["position"].shape == (2,)


def test_position_uses_per_coordinate_bounds():
    # Non-square map: a col value >= height must still be admissible (it is only
    # bounded by width), which a shared max(height, width) bound would also allow,
    # but a value >= width must be rejected -- proving the axes are independent.
    env, _ = _env(width=10, height=8)
    space = env.observation_space("r0")["position"]
    assert np.array_equal(space.high, np.array([7, 9]))  # (height-1, width-1)
    assert space.contains(np.array([7, 9], dtype=np.int64))
    assert not space.contains(np.array([9, 9], dtype=np.int64))  # row past height


def test_no_memory_sharing_with_live_belief():
    env, obs = _env()
    belief = obs["r0"]["belief_map"]
    robot_belief = env.robots["r0"].belief_map
    assert belief is not robot_belief
    assert belief.base is not robot_belief
    # Mutating the returned observation must not touch the live robot state.
    if belief.flags.writeable:
        belief[0, 0] = 1
        assert env.robots["r0"].belief_map[0, 0] != 1 or robot_belief[0, 0] != belief[0, 0]


def test_old_observation_is_immutable_across_steps():
    env, obs = _env()
    snapshot = obs["r0"]["belief_map"].copy()
    old_obs = obs["r0"]
    rng = np.random.default_rng(1)
    policy = make_policy("move_toward_frontier_bfs")
    # Drive several steps; the robot senses new cells each tick.
    for _ in range(20):
        actions = {rid: policy.act(obs[rid], rng) for rid in env.agents}
        obs, *_ = env.step(actions)
    assert np.array_equal(old_obs["belief_map"], snapshot)
    # And the live belief has genuinely moved on, so the test isn't vacuous.
    assert not np.array_equal(env.robots["r0"].belief_map, snapshot)


def test_policies_operate_through_observation_boundary():
    # A policy must act given only the returned observation -- no env/robot/map.
    env, obs = _env()
    rng = np.random.default_rng(2)
    for name in ("move_random", "move_toward_frontier_bfs", "move_toward_unknown_bfs"):
        policy = make_policy(name)
        action = policy.act(obs["r0"], rng)
        assert isinstance(action, Action)
        assert action in Action.action_space()


def test_action_space_is_five_primitives():
    env, _ = _env()
    assert env.action_space("r0").n == 5


def test_adapter_owns_its_data():
    # RobotObservation.from_obs must copy, not alias, the source arrays.
    source = {
        "position": np.array([2, 3], dtype=np.int64),
        "belief_map": np.full((8, 10), UNKNOWN, dtype=np.int8),
    }
    adapter = RobotObservation.from_obs(source)
    source["belief_map"][0, 0] = KNOWN_FREE
    assert adapter.belief_map[0, 0] == UNKNOWN
    assert adapter.map_shape == (8, 10)  # derived from the belief array
    assert adapter.position == (2, 3)
