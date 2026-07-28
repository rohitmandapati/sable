
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping

import numpy as np

from actions import Action
from map import Map
from observations import RobotObservation
from robot import KNOWN_FREE, Position, Robot

from pettingzoo.utils.env import ParallelEnv

import functools
from gymnasium.spaces import Box, Dict, Discrete
from pettingzoo.utils.env import ParallelEnv

# Sensor footprint: the robot's own cell plus its four orthogonal neighbors.
_SENSOR_OFFSETS: tuple[Position, ...] = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))

Reward = int
Info = dict[str, object]

class Environment(ParallelEnv):
    
    agents: list[str] 
        
    def __init__(
        self,
        width: int,
        height: int,
        robot_ids: Iterable[str] = ("robot",),
        obstacle_density: float = 0.2,
        min_free_fraction: float = 0.3,
        max_ticks: int = 100_000,
    ) -> None:
        self.width = width
        self.height = height
        self.robot_ids: list[str] = list(robot_ids)
        if not self.robot_ids:
            raise ValueError("Environment needs at least one robot id")
        if len(set(self.robot_ids)) != len(self.robot_ids):
            raise ValueError("robot ids must be unique")
        self.obstacle_density = obstacle_density
        self.min_free_fraction = min_free_fraction
        self.max_ticks = max_ticks

        self.map: Map | None = None
        self.robots: dict[str, Robot] = {}
        self.tick_count = 0
        self.possible_agents = list(self.robot_ids)
        self.agents = [] # starts empty, might change now or change later during reset
        
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(5) # STAY, UP, DOWN, LEFT, RIGHT, will be changed to communication actions
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Dict({
            "position":   Box(0, max(self.height, self.width), (2,), dtype=np.int64),
            "belief_map": Box(-1, 1, (self.height, self.width), dtype=np.int8),
        })

    # -- lifecycle -----------------------------------------------------------

    def reset(self, seed=None, options=None):
        # Build a fresh world for seed, spawn robots, sense, return observations.
        self.map = Map(
            width=self.width,
            height=self.height,
            seed=seed,
            obstacle_density=self.obstacle_density,
            min_free_fraction=self.min_free_fraction,
        )
        free_cells = self.map.free_cells
        if len(free_cells) < len(self.robot_ids):
            raise ValueError("Not enough free cells to spawn all robots")

        # Spawn from the map's own rng stream (same draw the pre-refactor code
        # used, so single-robot spawns are unchanged). Fully separating the
        # spawn rng from map generation is deferred; see project notes.
        indices = self.map.rng.choice(
            len(free_cells), size=len(self.robot_ids), replace=False
        )
        map_shape = (self.height, self.width)
        self.robots = {}
        for rid, idx in zip(self.robot_ids, np.atleast_1d(indices)):
            spawn = (int(free_cells[idx][0]), int(free_cells[idx][1]))
            self.robots[rid] = Robot(robot_id=rid, pos=spawn, map_shape=map_shape)

        self.tick_count = 0
        for robot in self.robots.values():
            self._sense(robot)
        self.agents = self.active_robot_ids()
        observations = self._observations()
        infos = {rid: {} for rid in self.agents}
        return observations, infos

    def step(self, actions: Mapping[str, object]) -> tuple[dict[str, RobotObservation], dict[str, Reward], bool, bool, Info]:
        if self.map is None:
            raise RuntimeError("call reset() before step()")

        self.tick_count += 1

        # Validate each alive robot's action into a target cell
        desired: dict[str, Position] = {}
        for rid, robot in self.robots.items():
            if not robot.alive:
                continue
            action = Action.coerce(actions.get(rid)) or Action.STAY
            desired[rid] = self._validated_target(robot, action)

        # Resolve simultaneous movement (single robot: no-op)
        resolved = self._resolve_collisions(desired)

        # Apply moves and sense, set_position appends to the trajectory every
        # tick
        newly: dict[str, int] = {}
        for rid, robot in self.robots.items():
            if not robot.alive:
                continue
            robot.set_position(resolved[rid])
            newly[rid] = self._sense(robot)

        raw = self._observations()
        terminated = self.coverage_complete()
        truncated = self.tick_count >= self.max_ticks and not terminated
        observations = {rid: raw[rid] for rid in self.agents}
        rewards = {rid: float(newly.get(rid, 0)) for rid in self.agents}
        terminations = {rid: terminated for rid in self.agents}
        truncations = {rid: truncated for rid in self.agents}
        infos = {rid: {} for rid in self.agents}
        
        self.agents = [rid for rid in self.agents if not (terminations[rid] or truncations[rid])]
        return observations, rewards, terminations, truncations, infos

    # -- queries -------------------------------------------------------------

    def active_robot_ids(self) -> list[str]:
        return [rid for rid, robot in self.robots.items() if robot.alive]

    def coverage(self) -> float:
        """Fraction of ground-truth free cells known-free to at least one robot."""
        assert self.map is not None
        free = self.map.free_cells
        if len(free) == 0:
            return 1.0
        known = self._known_free_mask()
        seen = sum(1 for r, c in free if known[r, c])
        return seen / len(free)

    def coverage_complete(self) -> bool:
        assert self.map is not None
        free = self.map.free_cells
        known = self._known_free_mask()
        return all(known[r, c] for r, c in free)

    # -- internals -----------------------------------------------------------

    def _observations(self) -> dict[str, RobotObservation]:
        return {
            rid: RobotObservation.from_robot(robot)
            for rid, robot in self.robots.items()
        }

    def _validated_target(self, robot: Robot, action: Action) -> Position:
        assert self.map is not None
        dr, dc = action.delta
        r, c = robot.pos
        nr, nc = r + dr, c + dc
        if not (0 <= nr < self.height and 0 <= nc < self.width):
            return robot.pos  # out of bounds -> stay
        if self.map.grid[nr, nc] == 1:
            return robot.pos  # walk into a wall -> stay
        return (nr, nc)

    def _resolve_collisions(
        self, desired: dict[str, Position]
    ) -> dict[str, Position]:
        
        current = {rid: self.robots[rid].pos for rid in desired}
        final = dict(desired)

        occupancy: dict[Position, list[str]] = defaultdict(list)
        for rid, target in desired.items():
            occupancy[target].append(rid)
        for target, rids in occupancy.items():
            if len(rids) > 1:
                for rid in rids:
                    final[rid] = current[rid]

        ids = sorted(desired)
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                if (
                    current[a] != current[b]
                    and final[a] == current[b]
                    and final[b] == current[a]
                ):
                    final[a] = current[a]
                    final[b] = current[b]
        return final

    def _sense(self, robot: Robot) -> int:
        """Reveal ground truth in the sensor footprint; return newly revealed."""
        assert self.map is not None
        row, col = robot.pos
        revealed = 0
        for dr, dc in _SENSOR_OFFSETS:
            r, c = row + dr, col + dc
            if 0 <= r < self.height and 0 <= c < self.width:
                if robot.belief_map[r, c] == -1:  # UNKNOWN
                    revealed += 1
                robot.reveal_cell((r, c), int(self.map.grid[r, c]))
        return revealed

    def _known_free_mask(self) -> np.ndarray:
        mask = np.zeros((self.height, self.width), dtype=bool)
        for robot in self.robots.values():
            mask |= robot.belief_map == KNOWN_FREE
        return mask
