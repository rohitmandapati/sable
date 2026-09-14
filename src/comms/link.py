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

from dataclasses import dataclass

import numpy as np

from comms.config import CommsConfig
from robot import Position


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
    # honours only delay 0 until the latency stage lands.
    delivered: bool
    delay_ticks: int = 0


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
        # Uniform Bernoulli loss. The RNG is drawn once iff drop_prob > 0 (the
        # short-circuit), so a lossless link consumes no randomness -- preserving
        # the exact draw order of the previous should_drop() implementation, and
        # thus byte-for-byte identical results.
        p = self.config.drop_prob
        if p > 0.0 and self._rng.random() < p:
            return LinkOutcome(delivered=False)
        return LinkOutcome(delivered=True)
