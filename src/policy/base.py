
from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

import numpy as np

from actions import Action
from observations import RobotObservation

PolicyFn = Callable[[RobotObservation, np.random.Generator], Action]


@runtime_checkable
class Policy(Protocol):
    def act(
        self,
        observation: RobotObservation,
        rng: np.random.Generator,
    ) -> Action:
        ...


class FunctionPolicy:
    """Adapts a `(observation, rng) -> Action` function to the Policy protocol."""

    def __init__(self, fn: PolicyFn, name: str | None = None) -> None:
        self._fn = fn
        self.name = name or getattr(fn, "__name__", "policy")

    def act(
        self,
        observation: RobotObservation,
        rng: np.random.Generator,
    ) -> Action:
        return self._fn(observation, rng)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"FunctionPolicy({self.name})"
