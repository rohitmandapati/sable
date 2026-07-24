# Turn a planned path into a primitive action.

from __future__ import annotations

from actions import Action

Cell = tuple[int, int]


def first_step_action(start: Cell, next_cell: Cell) -> Action:
    """The Action that moves from `start` to the adjacent `next_cell`."""
    return Action.from_delta((next_cell[0] - start[0], next_cell[1] - start[1]))
