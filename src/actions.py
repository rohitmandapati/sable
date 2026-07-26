# Actions a robot can take in the environment

from __future__ import annotations

from enum import Enum


class Action(Enum):
    STAY = (0, 0)
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)
    
    @classmethod
    def action_space(cls) -> set["Action"]:
        return {
            cls.STAY,
            cls.UP,
            cls.DOWN,
            cls.LEFT,
            cls.RIGHT,
        }

    @property
    def delta(self) -> tuple[int, int]:
        return self.value

    @classmethod
    def from_delta(cls, delta: tuple[int, int]) -> "Action":
        return _DELTA_TO_ACTION[(int(delta[0]), int(delta[1]))]

    @classmethod
    def coerce(cls, value: object) -> "Action | None":
        if isinstance(value, cls):
            return value
        if isinstance(value, tuple) and len(value) == 2:
            try:
                return _DELTA_TO_ACTION[(int(value[0]), int(value[1]))]
            except (KeyError, TypeError, ValueError):
                return None
        return None


_DELTA_TO_ACTION: dict[tuple[int, int], Action] = {a.value: a for a in Action}
