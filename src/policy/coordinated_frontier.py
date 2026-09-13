# Decentralized coordinated frontier assignment: the classical multi-robot
# exploration ceiling.
#
# There is no central coordinator. Every robot independently runs the SAME
# assignment (greedy or Hungarian) and then executes only its own share. Each
# robot reasons from:
#   * its OWN belief map (comms-limited) for frontier detection and all path
#     costs -- so a message dropped by the comms layer genuinely shrinks what
#     that robot can see and pursue; and
#   * every teammate's POSITION, which is assumed known (shared localization,
#     separate from the comms-limited belief).
# So a robot's action is a pure function of (its own belief, teammate positions)
# -- true decentralized execution. Under perfect comms every robot's belief is
# identical, so all robots compute the identical joint assignment and act
# consistently (matching a central coordinator). Under degraded comms their
# views diverge and they may double up -- which is the realistic behaviour we
# want to measure and, later, hand to a learned comms policy.

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from actions import Action
from observations import RobotObservation
from planning import (
    bfs_distance_field,
    cluster_frontiers,
    find_frontier_cells,
    first_step_action,
    shortest_path_to_any,
    shortest_path_to_any_astar,
)

Cell = tuple[int, int]

# Cost used for a robot-region pair that is unreachable, or for the dummy columns
# that pad the Hungarian cost matrix to square. Large enough to never win a
# min-cost assignment against any real (finite) path cost.
_UNREACHABLE = 1 << 30


class CoordinatedFrontierPolicy:
    def __init__(
        self,
        mode: str = "greedy",
        *,
        discount_radius: int = 0,
        path: str = "astar",
        name: str | None = None,
    ) -> None:
        if mode not in ("greedy", "hungarian"):
            raise ValueError(f"mode must be 'greedy' or 'hungarian', got {mode!r}")
        if path not in ("astar", "bfs"):
            raise ValueError(f"path must be 'astar' or 'bfs', got {path!r}")
        self.mode = mode
        # Manhattan radius (in cells) around an assigned region whose other
        # regions have their utility zeroed, so a second robot won't pick a
        # neighbour of an already-claimed region. 0 == region-level dedup only
        # (each connected region assigned to at most one robot). Greedy only.
        self.discount_radius = discount_radius
        self._path_to_any = (
            shortest_path_to_any_astar if path == "astar" else shortest_path_to_any
        )
        self.name = name or f"coordinated_frontier_{mode}"

    # -- Policy protocol -----------------------------------------------------

    def act(self, observation, rng: np.random.Generator) -> Action:
        # Single-robot convenience: coordination is a no-op with one robot.
        return self.act_joint({"_": observation}, {"_": rng})["_"]

    def act_joint(
        self,
        observations: Mapping[str, object],
        rngs: Mapping[str, np.random.Generator],
    ) -> dict[str, Action]:
        obs = {
            rid: o if isinstance(o, RobotObservation) else RobotObservation.from_obs(o)
            for rid, o in observations.items()
        }
        ids = list(obs)
        if not ids:
            return {}

        # Teammate positions are shared (localization); beliefs are not (they are
        # comms-limited and read per robot inside _decide). Each robot decides
        # independently -- no coordinator, no pooled belief.
        positions = {rid: obs[rid].position for rid in ids}
        return {
            rid: self._decide(rid, ids, positions, obs[rid].belief_map)
            for rid in ids
        }

    def _decide(
        self,
        me: str,
        ids: list[str],
        positions: Mapping[str, Cell],
        belief: np.ndarray,
    ) -> Action:
        # Robot `me`'s own view: frontiers, regions and every path cost are
        # computed on ITS belief alone. It runs the full team assignment (so it
        # knows which region is "its" share) but returns only its own action.
        frontiers = find_frontier_cells(belief)
        if not frontiers:
            return Action.STAY
        regions = cluster_frontiers(frontiers)

        # One BFS per robot over THIS belief -> region cost is the cheapest cell.
        # Under perfect comms every robot's belief is identical, so every robot
        # builds the same matrix and reaches the same assignment.
        cost = np.full((len(ids), len(regions)), float(_UNREACHABLE))
        for i, rid in enumerate(ids):
            field = bfs_distance_field(positions[rid], belief)
            for j, region in enumerate(regions):
                reachable = [field[c] for c in region if c in field]
                if reachable:
                    cost[i, j] = float(min(reachable))

        if self.mode == "greedy":
            assigned = self._assign_greedy(cost, regions)
        else:
            assigned = self._assign_hungarian(cost)

        my_idx = ids.index(me)
        j = assigned[my_idx]
        # No region for me (fewer regions than robots, or all unreachable from my
        # view): fall back to my own nearest reachable region -- doubling up beats
        # standing still.
        if j < 0 or cost[my_idx, j] >= _UNREACHABLE:
            j = int(np.argmin(cost[my_idx])) if cost[my_idx].min() < _UNREACHABLE else -1
        return self._step_toward(positions[me], regions, j, belief)

    # -- assignment rules ----------------------------------------------------

    def _assign_greedy(self, cost: np.ndarray, regions: list[set[Cell]]) -> list[int]:
        # Burgard cost-utility: repeatedly take the globally cheapest reachable
        # (robot, region) pair whose region still has utility, assign it, then
        # zero the utility of regions within discount_radius of it. Deterministic:
        # ties break by (cost, robot index, region index).
        n, m = cost.shape
        reps = [min(region) for region in regions]
        assigned = [-1] * n
        robot_free = [True] * n
        region_util = [True] * m

        for _ in range(min(n, m)):
            best = None  # (cost, i, j)
            for i in range(n):
                if not robot_free[i]:
                    continue
                for j in range(m):
                    if not region_util[j] or cost[i, j] >= _UNREACHABLE:
                        continue
                    key = (cost[i, j], i, j)
                    if best is None or key < best:
                        best = key
            if best is None:
                break
            _, i, j = best
            assigned[i] = j
            robot_free[i] = False
            rr, rc = reps[j]
            for k in range(m):
                if region_util[k]:
                    kr, kc = reps[k]
                    if abs(kr - rr) + abs(kc - rc) <= self.discount_radius:
                        region_util[k] = False
        return assigned

    def _assign_hungarian(self, cost: np.ndarray) -> list[int]:
        # Optimal min-total-cost matching of robots -> regions. Pad to a square
        # matrix so #robots != #regions is handled; padded/unreachable picks are
        # sorted out by the fallback in act_joint.
        n, m = cost.shape
        size = max(n, m)
        padded = np.full((size, size), float(_UNREACHABLE))
        padded[:n, :m] = cost
        col_for_row = _hungarian(padded)
        return [col_for_row[i] if col_for_row[i] < m else -1 for i in range(n)]

    # -- helpers -------------------------------------------------------------

    def _step_toward(
        self, pos: Cell, regions: list[set[Cell]], j: int, belief: np.ndarray
    ) -> Action:
        if j < 0:
            return Action.STAY
        path = self._path_to_any(pos, regions[j], belief)
        if path is None or len(path) < 2:
            return Action.STAY
        return first_step_action(path[0], path[1])

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"CoordinatedFrontierPolicy({self.mode})"


def _hungarian(cost: np.ndarray) -> list[int]:
    # Optimal square assignment (Kuhn-Munkres, O(n^3)); returns col per row.

    # Classic potentials/augmenting-path formulation for the min-cost assignment
    # problem. cost must be square; cost[i][j] is the cost of row i -> col j.

    n = cost.shape[0]
    INF = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)   # p[j] = row matched to column j (1-indexed; 0 = none)
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = float(cost[i0 - 1, j - 1]) - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0 != 0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    col_for_row = [-1] * n
    for j in range(1, n + 1):
        if p[j] != 0:
            col_for_row[p[j] - 1] = j - 1
    return col_for_row
