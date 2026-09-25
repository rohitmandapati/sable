# Coordinated frontier assignment: greedy (Burgard cost-utility) and Hungarian
# (optimal matching). The invariant that matters is that a belief-sharing team
# splits across distinct frontier regions instead of piling onto the nearest one.

import numpy as np

from actions import Action
from observations import RobotObservation
from policy import make_policy
from policy.coordinated_frontier import CoordinatedFrontierPolicy, _hungarian
from robot import KNOWN_FREE, UNKNOWN


def _belief_from(grid: np.ndarray) -> np.ndarray:
    b = grid.astype(np.int8)
    b.flags.writeable = False
    return b


def _obs(pos, belief):
    return RobotObservation(position=pos, belief_map=belief, alive=True)


def test_hungarian_matches_known_optimum():
    # Classic 3x3 assignment; optimal matching has total cost 5 (0->1,1->0,2->2).
    cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]], dtype=float)
    assign = _hungarian(cost)
    total = sum(cost[i, assign[i]] for i in range(3))
    assert total == 5
    assert sorted(assign) == [0, 1, 2]  # a permutation


def test_two_robots_split_across_two_regions():
    # A known-free corridor with an unknown gap at each end -> two frontier
    # regions (index 1 on the left, index 9 on the right). BOTH robots start on
    # the left side, so an uncoordinated team would send both to the near-left
    # frontier. Coordination must split them: one left, one right.
    row = np.full((1, 11), KNOWN_FREE, dtype=np.int8)
    row[0, 0] = UNKNOWN
    row[0, 10] = UNKNOWN
    belief = _belief_from(row)

    for mode in ("greedy", "hungarian"):
        policy = CoordinatedFrontierPolicy(mode=mode)
        obs = {"A": _obs((0, 2), belief), "B": _obs((0, 3), belief)}
        rngs = {rid: np.random.default_rng(0) for rid in obs}
        actions = policy.act_joint(obs, rngs)
        # A (nearer the left frontier) takes it; B is pushed to the right one.
        assert actions["A"] == Action.LEFT, mode
        assert actions["B"] == Action.RIGHT, mode


def test_decision_uses_own_belief_not_teammates():
    # Decentralized: a robot plans on ITS OWN belief. Robot B's belief has no
    # frontier at all (fully known local patch), so B must STAY even though
    # teammate A sits next to an unknown region A can see. B's ignorance (a
    # dropped message, say) genuinely limits B -- no pooled/global map leaks in.
    full_known = _belief_from(np.full((1, 5), KNOWN_FREE, dtype=np.int8))  # no UNKNOWN

    a_row = np.full((1, 5), KNOWN_FREE, dtype=np.int8)
    a_row[0, 0] = UNKNOWN  # A knows about a frontier at index 1
    a_belief = _belief_from(a_row)

    policy = CoordinatedFrontierPolicy(mode="greedy")
    obs = {"A": _obs((0, 2), a_belief), "B": _obs((0, 3), full_known)}
    rngs = {rid: np.random.default_rng(0) for rid in obs}
    actions = policy.act_joint(obs, rngs)

    assert actions["A"] == Action.LEFT   # A pursues the frontier it knows
    assert actions["B"] == Action.STAY   # B sees no frontier in its own belief


def test_identical_beliefs_reach_consistent_assignment():
    # Under perfect comms both robots share one belief -> each independently
    # computes the SAME joint assignment, so they split without a coordinator.
    row = np.full((1, 11), KNOWN_FREE, dtype=np.int8)
    row[0, 0] = UNKNOWN
    row[0, 10] = UNKNOWN
    belief = _belief_from(row)
    for mode in ("greedy", "hungarian"):
        policy = CoordinatedFrontierPolicy(mode=mode)
        obs = {"A": _obs((0, 2), belief), "B": _obs((0, 3), belief)}
        actions = policy.act_joint(obs, {r: np.random.default_rng(0) for r in obs})
        assert {actions["A"], actions["B"]} == {Action.LEFT, Action.RIGHT}, mode


def test_no_frontiers_all_stay():
    belief = _belief_from(np.full((3, 3), KNOWN_FREE, dtype=np.int8))
    policy = CoordinatedFrontierPolicy(mode="greedy")
    obs = {"a": _obs((1, 1), belief)}
    actions = policy.act_joint(obs, {"a": np.random.default_rng(0)})
    assert actions["a"] == Action.STAY


def test_registered_names_build():
    for name in ("coordinated_frontier_greedy", "coordinated_frontier_hungarian"):
        policy = make_policy(name)
        assert hasattr(policy, "act_joint")
