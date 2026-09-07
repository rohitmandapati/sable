
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping

import numpy as np

from actions import Action
from comms import Cell, CommsChannel, LinkModel
from map import Map
from robot import KNOWN_FREE, Position, Robot
from seeding import (
    EpisodeSeeds,
    derive_episode_seeds,
    validate_attempts,
    validate_dimension,
    validate_fraction,
)

from pettingzoo.utils.env import ParallelEnv

import functools
from gymnasium.spaces import Box, Dict, Discrete
from pettingzoo.utils.env import ParallelEnv

# Sensor footprint: the robot's own cell plus its four orthogonal neighbors.
_SENSOR_OFFSETS: tuple[Position, ...] = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))

Reward = int
Info = dict[str, object]


def resolve_moves(
    current: Mapping[str, Position],
    desired: Mapping[str, Position],
) -> tuple[dict[str, Position], set[str]]:
    """Resolve simultaneous one-step moves into unique, collision-free targets.

    Semantics are deterministic and independent of iteration order:

    * A robot requesting its own cell stays.
    * If two or more robots request the same destination, *all* of them are
      rejected (they stay) -- a destination is never shared.
    * A robot may move only if its destination is empty, or is vacated this tick
      by another robot that itself successfully moves. A chain of robots shifting
      into cells simultaneously vacated therefore succeeds only when it
      terminates in a genuinely empty cell; a chain terminating at a stationary
      or blocked robot is rejected backward along its whole length.
    * Cycles never reach an empty cell, so every cycle is rejected. This subsumes
      direct two-robot swaps (a 2-cycle) and larger rotations.

    Cells left in place by a wall / out-of-bounds request already arrive here as
    ``desired == current`` (the caller validates targets first), so they are not
    treated as movement.

    Returns ``(final, conflicted)`` where ``final`` maps each robot to a unique,
    free cell and ``conflicted`` is the set of robots that *wanted* to move
    (``desired != current``) but were held in place by a conflict.
    """
    ids = list(desired)
    # Which robot currently sits on each cell (its potential vacancy this tick).
    occupant = {current[r]: r for r in ids}

    movers = {r for r in ids if desired[r] != current[r]}

    # Same-destination contention: reject every robot targeting a shared cell.
    per_target: dict[Position, list[str]] = defaultdict(list)
    for r in movers:
        per_target[desired[r]].append(r)
    contested = {r for rs in per_target.values() if len(rs) > 1 for r in rs}
    candidates = movers - contested

    # A candidate can move iff its destination is empty or its destination's
    # occupant is itself a moving candidate. Grow ``valid`` from candidates whose
    # destination is already empty, propagating along vacated chains. The step is
    # monotone (``valid`` only grows) so the fixpoint -- and thus the result --
    # is the same regardless of the order robots are visited in.
    valid: set[str] = set()
    changed = True
    while changed:
        changed = False
        for r in candidates:
            if r in valid:
                continue
            occ = occupant.get(desired[r])
            if occ is None or occ in valid:
                valid.add(r)
                changed = True

    final = {r: (desired[r] if r in valid else current[r]) for r in ids}
    conflicted = movers - valid
    return final, conflicted

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
        max_generation_attempts: int = 100,
        map_name: str | None = None,
        enable_comms: bool = False,
        comms_drop_prob: float = 0.0,
        comms_max_bytes_per_tick: int | None = None,
        comms_seed: int | None = None,
    ) -> None:
        # Validate world inputs up front (Map re-validates at generation time).
        self.width = validate_dimension("width", width)
        self.height = validate_dimension("height", height)
        self.obstacle_density = validate_fraction("obstacle_density", obstacle_density)
        self.min_free_fraction = validate_fraction("min_free_fraction", min_free_fraction)
        self.max_generation_attempts = validate_attempts(
            "max_generation_attempts", max_generation_attempts
        )
        self.robot_ids: list[str] = list(robot_ids)
        if not self.robot_ids:
            raise ValueError("Environment needs at least one robot id")
        if len(set(self.robot_ids)) != len(self.robot_ids):
            raise ValueError("robot ids must be unique")
        self.max_ticks = max_ticks
        # Default named map (an explicit selection path, never the PettingZoo
        # seed); reset(options={"map": ...}) can override it per episode.
        self.map_name = map_name

        # Comms (off by default so the classical baseline is untouched). When on,
        # robots broadcast newly-sensed cells each tick; received cells fill
        # belief_map only (never sensed_mask), so physical-sensing redundancy
        # stays a true measure of duplicated exploration effort.
        self.enable_comms = enable_comms
        self.comms_drop_prob = comms_drop_prob
        self.comms_max_bytes_per_tick = comms_max_bytes_per_tick
        # An explicit comms seed pins that one stream regardless of the episode
        # root (so a caller can hold the map fixed while varying network loss).
        self.comms_seed = comms_seed
        self.comms: CommsChannel | None = None

        self.map: Map | None = None
        self.robots: dict[str, Robot] = {}
        self.tick_count = 0
        self.possible_agents = list(self.robot_ids)
        self.agents = [] # starts empty, might change now or change later during reset

        # Per-episode seed manifest (filled in reset), plus the independent
        # streams the environment itself owns.
        self.seeds: EpisodeSeeds | None = None
        self.active_map_name: str | None = None
        self._spawn_rng: np.random.Generator | None = None
        self._dynamics_rng: np.random.Generator | None = None

        # Cumulative movement-outcome counters for the episode, so the runner can
        # report friction the physics can't hide: actions rejected by a wall or
        # the map edge, and moves held in place by robot-robot conflicts.
        self.wall_bumps = 0
        self.conflicts = 0

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(5) # STAY, UP, DOWN, LEFT, RIGHT, will be changed to communication actions
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # Position is (row, col): row is bounded by height, column by width, so
        # the two coordinates get independent per-axis bounds (not a shared max).
        return Dict({
            "position":   Box(
                low=np.array([0, 0], dtype=np.int64),
                high=np.array([self.height - 1, self.width - 1], dtype=np.int64),
                shape=(2,),
                dtype=np.int64,
            ),
            "belief_map": Box(-1, 1, (self.height, self.width), dtype=np.int8),
        })

    # -- lifecycle -----------------------------------------------------------

    def reset(self, seed=None, options=None):
        # Build a fresh world for the episode, spawn robots, sense, return
        # observations. `seed` is the *root* episode seed (int or None); it is
        # split into independent map/spawn/dynamics/comms/policy streams. Named
        # default maps are chosen via options["map"] or the constructor's
        # map_name -- never by overloading `seed`.
        options = options or {}

        # Independent stream seeds. Per-stream overrides (constructor comms_seed
        # or options["seeds"]) let a caller hold the map fixed while varying,
        # e.g., spawn or network seeds.
        overrides: dict[str, int] = {}
        if self.comms_seed is not None:
            overrides["comms"] = self.comms_seed
        overrides.update(options.get("seeds", {}))
        self.seeds = derive_episode_seeds(seed, overrides=overrides)

        map_name = options.get("map", self.map_name)
        self.active_map_name = map_name
        if map_name is not None:
            # A named map establishes its own dimensions; validate they match the
            # environment the robots/spaces were configured for.
            self.map = Map(width=self.width, height=self.height, name=map_name)
            if self.map.grid.shape != (self.height, self.width):
                raise ValueError(
                    f"named map {map_name!r} is {self.map.grid.shape} but the "
                    f"environment is {(self.height, self.width)}; construct the "
                    f"Environment with matching width/height"
                )
        else:
            self.map = Map(
                width=self.width,
                height=self.height,
                seed=self.seeds.map,
                obstacle_density=self.obstacle_density,
                min_free_fraction=self.min_free_fraction,
                max_generation_attempts=self.max_generation_attempts,
            )

        self.wall_bumps = 0
        self.conflicts = 0

        # Independent RNG streams the environment owns
        self._spawn_rng = np.random.default_rng(self.seeds.spawn)
        self._dynamics_rng = np.random.default_rng(self.seeds.dynamics)

        # Fresh comms channel per episode, on its own seeded stream.
        if self.enable_comms:
            link = LinkModel(
                drop_prob=self.comms_drop_prob,
                max_bytes_per_tick=self.comms_max_bytes_per_tick,
                seed=self.seeds.comms,
            )
            self.comms = CommsChannel(link)
        else:
            self.comms = None

        free_cells = self.map.free_cells
        if len(free_cells) < len(self.robot_ids):
            raise ValueError("Not enough free cells to spawn all robots")

        # Spawn placement draws from its own stream, independent of map
        # generation, so the same map can be re-used with different spawns.
        indices = self._spawn_rng.choice(
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

    def episode_manifest(self) -> dict[str, object]:
        # Everything needed to replay the current episode.

        if self.seeds is None:
            raise RuntimeError("call reset() before episode_manifest()")
        return {
            "seeds": self.seeds.as_dict(),
            "map_name": self.active_map_name,
            "width": self.width,
            "height": self.height,
            "robot_ids": list(self.robot_ids),
            "requested_obstacle_density": (
                None if self.map is None else self.map.requested_obstacle_density
            ),
            "realized_obstacle_density": (
                None if self.map is None else self.map.realized_obstacle_density
            ),
        }

    def step(self, actions: Mapping[str, object]) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Reward], bool, bool, Info]:
        if self.map is None:
            raise RuntimeError("call reset() before step()")

        self.tick_count += 1

        # Validate each alive robot's action into a target cell. A move rejected
        # by a wall or the map edge collapses to "stay" here (target == pos); we
        # record it as a wall bump so the friction is observable.
        current: dict[str, Position] = {}
        desired: dict[str, Position] = {}
        wall_blocked: set[str] = set()
        for rid, robot in self.robots.items():
            if not robot.alive:
                continue
            current[rid] = robot.pos
            action = Action.coerce(actions.get(rid)) or Action.STAY
            target = self._validated_target(robot, action)
            if action is not Action.STAY and target == robot.pos:
                wall_blocked.add(rid)
            desired[rid] = target

        # Resolve simultaneous movement into unique, free cells (single robot:
        # no-op). ``conflicted`` are robots whose requested move lost to a
        # robot-robot conflict (same target, swap/cycle, or a blocked chain).
        resolved, conflicted = resolve_moves(current, desired)

        # Apply moves and sense, set_position appends to the trajectory every
        # tick
        newly: dict[str, int] = {}
        sensed_cells: dict[str, list[Cell]] = {}
        for rid, robot in self.robots.items():
            if not robot.alive:
                continue
            robot.set_position(resolved[rid])
            revealed = self._sense(robot)
            sensed_cells[rid] = revealed
            newly[rid] = len(revealed)

        # Share this tick's newly-sensed cells over the (lossy) comms channel and
        # fold whatever arrives into each robot's belief. Off unless enabled.
        if self.comms is not None:
            self._exchange_comms(sensed_cells)

        self.wall_bumps += len(wall_blocked)
        self.conflicts += len(conflicted)

        raw = self._observations()
        terminated = self.coverage_complete()
        truncated = self.tick_count >= self.max_ticks and not terminated
        observations = {rid: raw[rid] for rid in self.agents}
        rewards = {rid: float(newly.get(rid, 0)) for rid in self.agents}
        terminations = {rid: terminated for rid in self.agents}
        truncations = {rid: truncated for rid in self.agents}
        infos = {
            rid: {
                "wall_blocked": rid in wall_blocked,
                "blocked": rid in conflicted,
            }
            for rid in self.agents
        }

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

    def _observation(self, robot: Robot) -> dict[str, np.ndarray]:
        # A Gym-dict observation matching observation_space(agent). Arrays are
        # fresh copies, so each returned observation is a self-owned snapshot:
        # later mutation of the robot never changes an observation already handed
        # out. Policies read these only through the RobotObservation adapter.
        return {
            "position": np.array(robot.pos, dtype=np.int64),
            "belief_map": robot.belief_map.astype(np.int8, copy=True),
        }

    def _observations(self) -> dict[str, dict[str, np.ndarray]]:
        return {
            rid: self._observation(robot)
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

    def _sense(self, robot: Robot) -> list[Cell]:
        """Reveal ground truth in the sensor footprint; return newly-revealed cells.

        Each returned cell is ((row, col), belief_value) for a cell that was
        UNKNOWN before this sweep -- i.e. the belief delta this robot can share.
        """
        assert self.map is not None
        row, col = robot.pos
        revealed: list[Cell] = []
        for dr, dc in _SENSOR_OFFSETS:
            r, c = row + dr, col + dc
            if 0 <= r < self.height and 0 <= c < self.width:
                value = int(self.map.grid[r, c])
                if robot.belief_map[r, c] == -1:  # UNKNOWN -> newly revealed
                    revealed.append(((r, c), value))
                robot.reveal_cell((r, c), value)
                robot.sensed_mask[r, c] = True
        return revealed

    def _exchange_comms(self, sensed_cells: dict[str, list[Cell]]) -> None:
        # Broadcast each robot's newly-sensed cells to every other alive robot,
        # then drain inboxes into belief. Delivery is decided by the LinkModel
        # (uniform drop + per-recipient per-tick bandwidth cap); received cells
        # update belief_map ONLY -- reveal_cell fills UNKNOWN cells and never
        # touches sensed_mask, so a robot's first-hand sensing is never
        # overwritten and the redundancy metric stays physical.
        assert self.comms is not None
        alive = self.active_robot_ids()
        for rid in alive:
            cells = tuple(sensed_cells.get(rid, ()))
            if not cells:
                continue
            recipients = [other for other in alive if other != rid]
            if recipients:
                self.comms.send(rid, cells, recipients, self.tick_count)
        for rid in alive:
            robot = self.robots[rid]
            for message in self.comms.receive(rid):
                for (r, c), value in message.cells:
                    robot.reveal_cell((r, c), value)

    def _known_free_mask(self) -> np.ndarray:
        mask = np.zeros((self.height, self.width), dtype=bool)
        for robot in self.robots.values():
            mask |= robot.belief_map == KNOWN_FREE
        return mask

    def sensing_redundancy(self) -> float:
        # Average number of extra robots that physically sensed each cell.
        # rho = (sum over robots of |cells that robot sensed| - |team-sensed union|)
        #       / |team-sensed union|
        # Counts each robot's own-sensor coverage (free + wall), so it isolates
        # cross-robot duplicated exploration effort
        # 0 for a single robot; bounded above by n-1
        sum_sensed = 0
        union = np.zeros((self.height, self.width), dtype=bool)
        for robot in self.robots.values():
            sum_sensed += int(robot.sensed_mask.sum())
            union |= robot.sensed_mask
        team_sensed = int(union.sum())
        if team_sensed == 0:
            return 0.0
        return (sum_sensed - team_sensed) / team_sensed
