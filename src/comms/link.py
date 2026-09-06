# Transport-realism policy for the comms channel.
#
# Stage 1 of "realistic comms": uniform (distance-independent) message loss and a
# per-recipient, per-tick bandwidth cap. Delivery is still immediate -- only
# *whether* a message arrives changes here, not *when*.
#
# TODO (next stages): distance-based drop + hard range cutoff, then latency /
# staleness (delivery-tick gating in the channel's receive()).

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LinkModel:
    # Bernoulli loss applied independently per (message, recipient).
    drop_prob: float = 0.0
    # Per-recipient byte budget per tick; None means unlimited bandwidth.
    max_bytes_per_tick: int | None = None
    # Seed for this model's dedicated RNG stream (kept independent of the map and
    # collision RNGs so comms draws are reproducible on their own).
    seed: int | None = None

    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.drop_prob <= 1.0:
            raise ValueError(f"drop_prob must be in [0, 1], got {self.drop_prob}")
        if self.max_bytes_per_tick is not None and self.max_bytes_per_tick < 0:
            raise ValueError(
                f"max_bytes_per_tick must be >= 0, got {self.max_bytes_per_tick}"
            )
        self._rng = np.random.default_rng(self.seed)

    def reset(self, seed: int | None = None) -> None:
        # Re-seed at episode start so a re-run reproduces exactly the same results
        # Falls back to this model's configured seed when none is supplied
        self._rng = np.random.default_rng(self.seed if seed is None else seed)

    def should_drop(self) -> bool:
        # One uniform Bernoulli trial per call
        # the RNG is the only source of nondeterminism in the channel
        return self.drop_prob > 0.0 and bool(self._rng.random() < self.drop_prob)
