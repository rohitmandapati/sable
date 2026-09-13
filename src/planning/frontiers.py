# Frontier detection over a belief map
from __future__ import annotations

from collections import deque

import numpy as np

from robot import KNOWN_FREE, UNKNOWN

Cell = tuple[int, int]

_STEPS: tuple[Cell, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))


def is_frontier_cell(belief_map: np.ndarray, pos: Cell) -> bool:
    r, c = pos
    h, w = belief_map.shape
    for dr, dc in _STEPS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and belief_map[nr, nc] == UNKNOWN:
            return True
    return False


def find_frontier_cells(belief_map: np.ndarray) -> set[Cell]:
    free = np.argwhere(belief_map == KNOWN_FREE)
    return {
        (int(r), int(c))
        for r, c in free
        if is_frontier_cell(belief_map, (int(r), int(c)))
    }


def cluster_frontiers(frontier_cells: set[Cell]) -> list[set[Cell]]:
    # Group frontier cells into 4-connected regions (connected components).

    # Coordinated assignment operates on regions, not individual cells: two robots
    # handed adjacent cells of the same blob would still explore the same area, so
    # a whole region is one assignable target. Regions are returned sorted by their
    # minimum cell for deterministic downstream assignment.
    
    remaining = set(frontier_cells)
    clusters: list[set[Cell]] = []
    while remaining:
        seed = min(remaining)  # deterministic component order
        remaining.discard(seed)
        component: set[Cell] = {seed}
        queue: deque[Cell] = deque([seed])
        while queue:
            r, c = queue.popleft()
            for dr, dc in _STEPS:
                nb = (r + dr, c + dc)
                if nb in remaining:
                    remaining.discard(nb)
                    component.add(nb)
                    queue.append(nb)
        clusters.append(component)
    clusters.sort(key=min)
    return clusters
