# Primitives that back coordinated frontier assignment: a per-robot BFS distance
# field (costs to every reachable cell in one sweep) and frontier clustering
# (connected regions as single assignable targets).

import numpy as np

from planning import bfs_distance_field, cluster_frontiers
from robot import KNOWN_FREE, KNOWN_WALL, UNKNOWN


def test_distance_field_counts_steps_on_open_map():
    belief = np.full((3, 3), KNOWN_FREE, dtype=np.int8)
    dist = bfs_distance_field((0, 0), belief)
    assert dist[(0, 0)] == 0
    assert dist[(0, 1)] == 1
    assert dist[(1, 0)] == 1
    # Manhattan distance on an obstacle-free grid (4-connectivity).
    assert dist[(2, 2)] == 4


def test_distance_field_routes_around_walls():
    # A wall column splits the map; the only gap is the bottom row.
    belief = np.full((3, 3), KNOWN_FREE, dtype=np.int8)
    belief[0, 1] = KNOWN_WALL
    belief[1, 1] = KNOWN_WALL
    dist = bfs_distance_field((0, 0), belief)
    # (0,2) is not reachable straight across; must detour down and back up.
    assert dist[(0, 2)] == 6
    # Known walls are never entered, so they carry no distance.
    assert (0, 1) not in dist


def test_distance_field_treats_unknown_as_passable():
    # Planning is optimistic: UNKNOWN cells may be traversable, so they get a
    # distance (only KNOWN_WALL blocks).
    belief = np.full((2, 2), UNKNOWN, dtype=np.int8)
    belief[0, 0] = KNOWN_FREE
    dist = bfs_distance_field((0, 0), belief)
    assert dist[(1, 1)] == 2


def test_distance_field_omits_unreachable_cells():
    belief = np.full((1, 3), KNOWN_FREE, dtype=np.int8)
    belief[0, 1] = KNOWN_WALL  # walls off the right cell entirely
    dist = bfs_distance_field((0, 0), belief)
    assert (0, 2) not in dist


def test_cluster_frontiers_groups_connected_cells():
    cells = {(0, 0), (0, 1), (1, 0), (5, 5), (5, 6)}
    clusters = cluster_frontiers(cells)
    assert len(clusters) == 2
    assert {(0, 0), (0, 1), (1, 0)} in clusters
    assert {(5, 5), (5, 6)} in clusters


def test_cluster_frontiers_diagonal_cells_are_separate():
    # 4-connectivity: diagonal neighbors are NOT the same region.
    clusters = cluster_frontiers({(0, 0), (1, 1)})
    assert len(clusters) == 2


def test_cluster_frontiers_is_deterministic_and_sorted():
    cells = {(3, 3), (0, 0), (0, 1)}
    clusters = cluster_frontiers(cells)
    # Sorted by minimum cell: the (0,0) region comes before the (3,3) region.
    assert [min(c) for c in clusters] == [(0, 0), (3, 3)]


def test_cluster_frontiers_empty():
    assert cluster_frontiers(set()) == []
