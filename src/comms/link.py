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
# TODO Longterm, replace with GoLang networking daemon.

from __future__ import annotations

import math
import zlib
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
    # Stable id of the message this decision is about; seeds the per-message
    # latency RNG. Defaults to 0 so unit callers that only exercise drop physics
    # need not supply one; the channel always passes the real id.
    message_id: int = 0


@dataclass(frozen=True)
class LinkOutcome:
    # The link's verdict for one (message, recipient). `delay_ticks` is the
    # latency the channel applies: the message is handed to the recipient
    # `delay_ticks` after it was sent (0 = same-tick delivery). `drop_cause`
    # labels a non-delivery for per-cause metrics (None when delivered).
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
        # Fixed entropy for the per-message latency RNG (independent of the drop
        # stream above). Captured once so re-runs with the same seed reproduce;
        # SeedSequence(None) draws and stores OS entropy, matching default_rng.
        self._latency_entropy = np.random.SeedSequence(seed).entropy

    @property
    def max_bytes_per_tick(self) -> int | None:
        # Exposed for the channel's bandwidth accounting.
        return self.config.max_bytes_per_tick

    def reset(self, seed: int | None = None) -> None:
        # Re-seed at episode start so a re-run reproduces exactly the same draws.
        # Falls back to this model's configured seed when none is supplied.
        s = self.seed if seed is None else seed
        self._rng = np.random.default_rng(s)
        self._latency_entropy = np.random.SeedSequence(s).entropy

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

        # Ground-truth walls on the line of sight, computed once and reused for
        # both occlusion loss and per-wall latency (skipped when neither needs it).
        wall_count = self._wall_count(a, b)
        p_dist = self._distance_drop(dist)
        p_occ = self._occlusion_drop(wall_count)
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

        delay = self._delay_ticks(query, dist, wall_count)
        return LinkOutcome(delivered=True, delay_ticks=delay)

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

    def _occlusion_drop(self, wall_count: int) -> float:
        # Added drop probability from ground-truth walls on the line of sight.
        atten = self.config.wall_attenuation
        if atten <= 0.0:
            return 0.0
        return min(1.0, atten * wall_count)

    def _wall_count(self, a: Position | None, b: Position | None) -> int:
        # Ground-truth walls between a and b, computed only when some cause
        # (occlusion drop or per-wall latency) actually consumes it -- otherwise
        # the Bresenham walk is skipped entirely.
        cfg = self.config
        if a is None or b is None or self.grid is None:
            return 0
        if cfg.wall_attenuation <= 0.0 and cfg.latency_per_wall <= 0.0:
            return 0
        return _wall_count_between(self.grid, a, b)

    # -- latency -------------------------------------------------------------

    def _delay_ticks(
        self, query: LinkQuery, dist: float | None, wall_count: int
    ) -> int:
        # Delay = physics-shaped mean + exponential jitter, rounded and clamped.
        # The no-latency fast path returns 0 without spinning any RNG, so an
        # unconfigured link is byte-for-byte identical to same-tick delivery.
        cfg = self.config
        mean = cfg.latency_base
        if dist is not None:
            mean += cfg.latency_per_distance * dist
        mean += cfg.latency_per_wall * wall_count
        if mean <= 0.0 and cfg.latency_jitter <= 0.0:
            return 0

        jitter = 0.0
        if cfg.latency_jitter > 0.0:
            rng_jitter, _reserved = self._latency_rng(
                query.message_id, query.recipient_id
            )
            jitter = float(rng_jitter.exponential(cfg.latency_jitter))

        delay = max(0, int(round(mean + jitter)))
        if cfg.max_latency_ticks is not None:
            delay = min(delay, cfg.max_latency_ticks)
        return delay

    def _latency_rng(
        self, message_id: int, recipient_id: str
    ) -> tuple[np.random.Generator, np.random.Generator]:
        # Two independent RNG streams derived from (link entropy, message id,
        # recipient) -- so a message's delay reproduces regardless of the order
        # evaluate() is called in, and without touching the drop stream. crc32
        # gives a stable, process-independent hash of the recipient id (Python's
        # built-in hash() is salted per run and would break reproducibility).
        # Stream two is reserved for the later deadzone stage.
        key = [self._latency_entropy, int(message_id), _stable_id_hash(recipient_id)]
        child_a, child_b = np.random.SeedSequence(key).spawn(2)
        return np.random.default_rng(child_a), np.random.default_rng(child_b)


def _stable_id_hash(recipient_id: str) -> int:
    return zlib.crc32(recipient_id.encode("utf-8"))


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
