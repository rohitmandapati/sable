# Transport-realism policy for the comms channel.
#
# The channel asks the LinkModel one question per (message, recipient):
# `evaluate(query) -> outcome`. The query carries everything a realistic link
# might need -- who is talking to whom, where they are, the tick, and the message
# size -- and the outcome says whether the message is delivered and (later) with
# what delay. Centralising the decision behind this one call is what lets later
# stages add distance path-loss, obstacle occlusion, a seed-driven noise field,
# latency and partial loss without touching the channel or the environment.
#
# Stage 1 (this file) implements only uniform (distance-independent) Bernoulli
# loss; the bandwidth cap lives in the channel because it is stateful per-tick
# accounting. Positions and grid are carried/stored but unused until the
# distance/occlusion stage. Defaults describe a perfect (lossless) link.
#
# TODO Longterm, replace with GoLang networking daemon.

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from comms.config import CommsConfig
from robot import KNOWN_WALL, Position


@dataclass(frozen=True)
class LinkQuery:
    # One delivery decision's worth of context: a single message from sender_id
    # to recipient_id. Positions are the robots' current cells (or None when a
    # caller does not supply them); the uniform stage ignores them.
    sender_id: str
    recipient_id: str
    sender_pos: Position | None
    recipient_pos: Position | None
    tick: int
    size_bytes: int


@dataclass(frozen=True)
class LinkOutcome:
    # The link's verdict for one (message, recipient). `delay_ticks` is the
    # designed seam for latency (0 = same-tick delivery, as today); the channel
    # honours only delay 0 until the latency stage lands. `drop_cause` labels a
    # non-delivery for per-cause metrics (None when delivered).
    delivered: bool
    delay_ticks: int = 0
    drop_cause: str | None = None


# Drop-cause labels (for metrics/attribution).
CAUSE_OUT_OF_RANGE = "out_of_range"
CAUSE_OCCLUDED = "occluded"
CAUSE_STOCHASTIC = "stochastic"  # combined uniform + distance + occlusion loss


class LinkModel:
    def __init__(
        self,
        config: CommsConfig | None = None,
        *,
        seed: int | None = None,
        grid: np.ndarray | None = None,
    ) -> None:
        self.config = config if config is not None else CommsConfig()
        self.seed = seed
        # Ground-truth grid the link operates over (env-side physics, never seen
        # by policies). Stored for the distance/occlusion stage; unused here.
        self.grid = grid
        self._rng = np.random.default_rng(seed)

    @property
    def max_bytes_per_tick(self) -> int | None:
        # Exposed for the channel's bandwidth accounting.
        return self.config.max_bytes_per_tick

    def reset(self, seed: int | None = None) -> None:
        # Re-seed at episode start so a re-run reproduces exactly the same draws.
        # Falls back to this model's configured seed when none is supplied.
        self._rng = np.random.default_rng(self.seed if seed is None else seed)

    def evaluate(self, query: LinkQuery) -> LinkOutcome:
        # Compose independent physical loss causes into one survival probability
        # and draw once. Deterministic hard failures (out of range, fully
        # occluded) short-circuit before any RNG draw and are attributed exactly.
        #
        # When no distance/occlusion applies (p_dist == p_occ == 0) the drop
        # probability is exactly config.drop_prob and the RNG is drawn once iff
        # that is > 0 -- byte-for-byte identical to the stage-1 uniform model.
        a, b = query.sender_pos, query.recipient_pos
        dist = _distance(a, b)

        # Hard range cutoff.
        cr = self.config.comm_range
        if cr is not None and dist is not None and dist >= cr:
            return LinkOutcome(delivered=False, drop_cause=CAUSE_OUT_OF_RANGE)

        p_dist = self._distance_drop(dist)
        p_occ = self._occlusion_drop(a, b)
        # A wall stack that fully attenuates is a deterministic block.
        if p_occ >= 1.0:
            return LinkOutcome(delivered=False, drop_cause=CAUSE_OCCLUDED)

        p_uniform = self.config.drop_prob
        if p_dist == 0.0 and p_occ == 0.0:
            p_drop = p_uniform  # exact stage-1 path (no float round-trip)
        else:
            p_drop = 1.0 - (1.0 - p_uniform) * (1.0 - p_dist) * (1.0 - p_occ)

        if p_drop > 0.0 and self._rng.random() < p_drop:
            return LinkOutcome(delivered=False, drop_cause=CAUSE_STOCHASTIC)
        return LinkOutcome(delivered=True)

    # -- physical loss causes ------------------------------------------------

    def _distance_drop(self, dist: float | None) -> float:
        # Path-loss between reliable_range and comm_range, 0 below, (approaching)
        # 1 at the cutoff. 0 when distance effects are disabled or unknown.
        cr = self.config.comm_range
        if cr is None or dist is None:
            return 0.0
        free = self.config.reliable_range
        if dist <= free:
            return 0.0
        # dist >= cr is handled as a hard cutoff by the caller; here dist < cr.
        return ((dist - free) / (cr - free)) ** self.config.distance_falloff

    def _occlusion_drop(self, a: Position | None, b: Position | None) -> float:
        # Added drop probability from ground-truth walls on the line of sight.
        atten = self.config.wall_attenuation
        if atten <= 0.0 or a is None or b is None or self.grid is None:
            return 0.0
        return min(1.0, atten * _wall_count_between(self.grid, a, b))


def _distance(a: Position | None, b: Position | None) -> float | None:
    if a is None or b is None:
        return None
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _wall_count_between(grid: np.ndarray, a: Position, b: Position) -> int:
    # Ground-truth wall cells strictly between a and b along a Bresenham line.
    # Endpoints (the robots' own free cells) are excluded.
    line = _bresenham(a, b)
    return sum(1 for (r, c) in line[1:-1] if grid[r, c] == KNOWN_WALL)


def _bresenham(a: Position, b: Position) -> list[Position]:
    # Integer line from a to b inclusive (row, col).
    r0, c0 = a
    r1, c1 = b
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    r, c = r0, c0
    cells: list[Position] = []
    while True:
        cells.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
    return cells
