
from __future__ import annotations

from typing import Callable, Mapping, Protocol, Union, runtime_checkable

import numpy as np

from actions import Action
from observations import RobotObservation

PolicyFn = Callable[[RobotObservation, np.random.Generator], Action]
Observation = Union[RobotObservation, Mapping[str, np.ndarray]]


@runtime_checkable
class Policy(Protocol):
    def act(
        self,
        observation: Observation,
        rng: np.random.Generator,
    ) -> Action:
        ...

    def act_joint(
        self,
        observations: Mapping[str, Observation],
        rngs: Mapping[str, np.random.Generator],
    ) -> dict[str, Action]:
        # Decide all robots' actions together. Independent policies just map
        # `act` over each robot; coordinated policies (e.g. frontier assignment)
        # use the whole set to hand out distinct targets. The runner always drives
        # policies through this one entry point.
        ...


class FunctionPolicy:
    """Adapts a `(observation, rng) -> Action` function to the Policy protocol."""

    def __init__(self, fn: PolicyFn, name: str | None = None) -> None:
        self._fn = fn
        self.name = name or getattr(fn, "__name__", "policy")

    def act(
        self,
        observation: Observation,
        rng: np.random.Generator,
    ) -> Action:
        # The environment returns Gym-dict observations; wrap them at the policy
        # boundary so classical policies keep their ergonomic attribute access.
        if not isinstance(observation, RobotObservation):
            observation = RobotObservation.from_obs(observation)
        return self._fn(observation, rng)

    def act_joint(
        self,
        observations: Mapping[str, Observation],
        rngs: Mapping[str, np.random.Generator],
    ) -> dict[str, Action]:
        # No coordination: each robot decides from its own observation alone.
        return {rid: self.act(observations[rid], rngs[rid]) for rid in observations}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"FunctionPolicy({self.name})"
