# The boundary between the simulator and policies

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # avoid a runtime import cycle; used for type hints only
    from robot import Robot


@dataclass(frozen=True, eq=False)
class RobotObservation:
    # What a policy sees for one robot on one tick

    robot_id: str
    position: tuple[int, int]
    belief_map: np.ndarray
    map_shape: tuple[int, int]
    alive: bool

    @classmethod
    def from_robot(cls, robot: "Robot") -> "RobotObservation":
        view = robot.belief_map.view()
        view.flags.writeable = False
        return cls(
            robot_id=robot.robot_id,
            position=(int(robot.pos[0]), int(robot.pos[1])),
            belief_map=view,
            map_shape=(int(robot.map_shape[0]), int(robot.map_shape[1])),
            alive=robot.alive,
        )
    
    def _to_gym_obs(self, obs):
        return {
            "position":   np.asarray(obs.position, dtype=np.int64),
            "belief_map": np.asarray(obs.belief_map, dtype=np.int8),
        }

