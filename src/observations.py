# The boundary between the simulator and policies
#
# The Environment (PettingZoo ParallelEnv) returns plain Gym-dict observations
# that satisfy observation_space(agent):
#
#     {"position": np.ndarray (2,) int64, "belief_map": np.ndarray (H, W) int8}


from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True, eq=False)
class RobotObservation:
    # What a policy sees for one robot on one tick.

    position: tuple[int, int]
    belief_map: np.ndarray  # owned, read-only int8 snapshot
    alive: bool = True
    robot_id: str | None = None

    @property
    def map_shape(self) -> tuple[int, int]:
        # Derived from the belief array so the two can never disagree.
        height, width = self.belief_map.shape
        return (int(height), int(width))

    @classmethod
    def from_obs(
        cls,
        obs: Mapping[str, np.ndarray],
        *,
        robot_id: str | None = None,
        alive: bool = True,
    ) -> "RobotObservation":
        # Wrap a Gym-dict observation returned by the environment.

        # Copies the belief map so this observation owns its data and is immune to
        # later mutation of the source array. The environment only ever returns
        # observations for live agents, so alive defaults to True.

        position = obs["position"]
        belief = np.array(obs["belief_map"], dtype=np.int8, copy=True)
        belief.flags.writeable = False
        return cls(
            position=(int(position[0]), int(position[1])),
            belief_map=belief,
            alive=alive,
            robot_id=robot_id,
        )
