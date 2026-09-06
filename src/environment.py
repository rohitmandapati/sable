
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping

import numpy as np

from actions import Action
from comms import Cell, CommsChannel, LinkModel
from map import Map
from observations import RobotObservation
from robot import KNOWN_FREE, Position, Robot

from pettingzoo.utils.env import ParallelEnv

import functools
from gymnasium.spaces import Box, Dict, Discrete
from pettingzoo.utils.env import ParallelEnv

# Sensor footprint: the robot's own cell plus its four orthogonal neighbors.
_SENSOR_OFFSETS: tuple[Position, ...] = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))

# Consecutive contested ticks a cell tolerates before randomized tie-breaking
# kicks in. After more than this many ticks of robots racing the same empty
# cell, one contender is chosen at random to proceed so the team can't livelock.
_CONTENTION_LIMIT = 2

# Stochastic backtracking: a robot that wanted to move but was held in place for
# more than this many consecutive ticks will, with probability _BACKTRACK_PROB,
# take a random legal step instead of its policy move. This perturbs a robot out
# of a standoff that tie-breaking alone can't resolve -- e.g. a robot endlessly
# re-targeting a cell it keeps losing and oscillating in place.
_BACKTRACK_LIMIT = 2
_BACKTRACK_PROB = 0.5

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
        enable_comms: bool = False,
        comms_drop_prob: float = 0.0,
        comms_max_bytes_per_tick: int | None = None,
        comms_seed: int | None = None,
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

        # Comms (off by default so the classical baseline is untouched). When on,
        # robots broadcast newly-sensed cells each tick; received cells fill
        # belief_map only (never sensed_mask), so physical-sensing redundancy
        # stays a true measure of duplicated exploration effort.
        self.enable_comms = enable_comms
        self.comms_drop_prob = comms_drop_prob
        self.comms_max_bytes_per_tick = comms_max_bytes_per_tick
        self.comms_seed = comms_seed
        self.comms: CommsChannel | None = None

        self.map: Map | None = None
        self.robots: dict[str, Robot] = {}
        self.tick_count = 0
        self.possible_agents = list(self.robot_ids)
        self.agents = [] # starts empty, might change now or change later during reset

        # Randomized tie-breaking state (set up per-episode in reset):
        # a dedicated RNG so collision draws are reproducible and independent of
        # the map stream, plus a per-cell counter of consecutive contested ticks.
        self._rng: np.random.Generator | None = None
        self._contention: dict[Position, int] = {}
        # Per-robot consecutive "blocked" tick counter, drives stochastic
        # backtracking (see _BACKTRACK_LIMIT).
        self._stuck: dict[str, int] = {}

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
        # Tie-break RNG is seeded from the episode seed (independent of the map's
        # own stream) so collision resolution is reproducible per run.
        self._rng = np.random.default_rng(seed)
        self._contention = {}
        self._stuck = {}

        # Fresh comms channel per episode. The link RNG is seeded from an explicit
        # comms_seed when given, else the episode seed, so runs reproduce.
        if self.enable_comms:
            link = LinkModel(
                drop_prob=self.comms_drop_prob,
                max_bytes_per_tick=self.comms_max_bytes_per_tick,
                seed=self.comms_seed if self.comms_seed is not None else seed,
            )
            self.comms = CommsChannel(link)
        else:
            self.comms = None

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
            target = self._validated_target(robot, action)
            if (
                self._stuck.get(rid, 0) > _BACKTRACK_LIMIT
                and self._rng.random() < _BACKTRACK_PROB
            ):
                target = self._random_step_target(robot)
            desired[rid] = target

        # Positions before movement, needed to tell whether a robot was blocked.
        before: dict[str, Position] = {rid: self.robots[rid].pos for rid in desired}

        # Resolve simultaneous movement (single robot: no-op)
        resolved = self._resolve_collisions(desired)

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

        # A robot that wanted to move but stayed put was blocked; track the run
        # of consecutive blocked ticks so backtracking can trigger.
        for rid in desired:
            if desired[rid] != before[rid] and resolved[rid] == before[rid]:
                self._stuck[rid] = self._stuck.get(rid, 0) + 1
            else:
                self._stuck[rid] = 0

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

    def _random_step_target(self, robot: Robot) -> Position:
        # A uniformly random in-bounds, non-wall neighbour (or stay if boxed in).
        # Collision resolution still applies, so this move is not privileged.
        assert self.map is not None and self._rng is not None
        r, c = robot.pos
        candidates = [
            (r + dr, c + dc)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if 0 <= r + dr < self.height
            and 0 <= c + dc < self.width
            and self.map.grid[r + dr, c + dc] == 0
        ]
        if not candidates:
            return robot.pos
        return candidates[int(self._rng.integers(len(candidates)))]

    def _resolve_collisions(
        self, desired: dict[str, Position]
    ) -> dict[str, Position]:
        
        current = {rid: self.robots[rid].pos for rid in desired}
        final = dict(desired)

        occupancy: dict[Position, list[str]] = defaultdict(list)
        for rid, target in desired.items():
            occupancy[target].append(rid)

        contested_now: set[Position] = set()
        for target, rids in occupancy.items():
            if len(rids) <= 1:
                continue

            # A robot already sitting on the target cell holds it; nobody may
            # move in on top of it (that would put two robots on one cell), so
            # the others just stay and re-route next tick (not a race)
            if any(current[rid] == target for rid in rids):
                for rid in rids:
                    if current[rid] != target:
                        final[rid] = current[rid]
                continue

            # every contender is moving into a currently-empty cell.
            # Count consecutive contested ticks; once it exceeds the limit,
            # pick one contender at random to proceed and hold the rest. Below
            # the limit, keep the conservative "everyone stays" rule.
            contested_now.add(target)
            self._contention[target] = self._contention.get(target, 0) + 1
            if self._contention[target] > _CONTENTION_LIMIT:
                assert self._rng is not None  # set in reset()
                # Sorted candidates keep the draw reproducible; which robot is
                # favoured is a policy knob we may learn/optimize later.
                winner = str(self._rng.choice(sorted(rids)))
                for rid in rids:
                    if rid != winner:
                        final[rid] = current[rid]
            else:
                for rid in rids:
                    final[rid] = current[rid]

        # Forget cells that were not contested this tick so the counter only
        # tracks *consecutive* standoffs.
        for target in [t for t in self._contention if t not in contested_now]:
            del self._contention[target]

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
