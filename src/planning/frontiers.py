# Frontier detection over a belief map
from __future__ import annotations

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
