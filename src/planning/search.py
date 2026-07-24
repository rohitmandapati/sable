# Reusable search over a belief map

from __future__ import annotations

import heapq
from collections import deque
from typing import Callable

import numpy as np

from robot import KNOWN_WALL

Cell = tuple[int, int]

#  expansion order: down, up, right, left
_STEPS: tuple[Cell, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))


def neighbors(pos: Cell, shape: tuple[int, int]) -> list[Cell]:
    r, c = pos
    h, w = shape
    out: list[Cell] = []
    for dr, dc in _STEPS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _default_passable(value: int) -> bool:
    # free and (optimistically) unknown cells; only known walls block.
    return value != KNOWN_WALL


def shortest_path_to_any(
    start: Cell,
    targets: set[Cell],
    belief_map: np.ndarray,
    passable: Callable[[int], bool] | None = None,
) -> list[Cell] | None:
    #BFS from start to the nearest cell in targets
    
    if passable is None:
        passable = _default_passable

    shape = (int(belief_map.shape[0]), int(belief_map.shape[1]))
    start = (int(start[0]), int(start[1]))

    parent: dict[Cell, Cell | None] = {start: None}
    queue: deque[Cell] = deque([start])

    while queue:
        current = queue.popleft()
        for nb in neighbors(current, shape):
            if nb in parent:
                continue
            parent[nb] = current
            if not passable(int(belief_map[nb])):
                continue
            if nb in targets:
                return _reconstruct(parent, nb)
            queue.append(nb)

    return None

def shortest_path_to_any_astar(
    start: Cell,
    targets: set[Cell],
    belief_map: np.ndarray,
    passable: Callable[[int], bool] | None = None,
) -> list[Cell] | None:
    #A* from start to the nearest cell in targets    
    if passable is None:
        passable = _default_passable
        
    shape = (int(belief_map.shape[0]), int(belief_map.shape[1]))
    start = (int(start[0]), int(start[1]))
    
    if not targets:
        return None

    parent: dict[Cell, Cell | None] = {start: None}
    g: dict[Cell, int] = {start: 0}          # cheapest known cost start -> cell
    queue = [(0, start)]                      # priority queue of (f = g + h, cell)

    # Manhattan distance to the nearest target
    h = lambda cell: min(abs(cell[0] - t[0]) + abs(cell[1] - t[1]) for t in targets)

    while queue:
        _, current = heapq.heappop(queue)
        if current in targets:
            return _reconstruct(parent, current)
        for nb in neighbors(current, shape):
            if not passable(int(belief_map[nb])):
                continue
            cost = g[current] + 1
            if cost < g.get(nb, 1 << 30):     # first time here, or a cheaper route
                g[nb] = cost
                parent[nb] = current
                heapq.heappush(queue, (cost + h(nb), nb))

    return None

def _reconstruct(parent: dict[Cell, Cell | None], end: Cell) -> list[Cell]:
    path = [end]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])  # type: ignore[arg-type]
    path.reverse()
    return path
