# Configuration for the communication channel's transport realism.
#
# One place for every comms knob, so the Environment/Runner take a single config
# object instead of a growing pile of scalar kwargs. Today it holds only the
# stage-1 knobs (uniform drop + a per-recipient bandwidth cap); later stages add
# fields here (distance path-loss + range, obstacle occlusion, a seed-driven
# noise field, latency, partial/byte loss) and the LinkModel reads them. Defaults
# describe a perfect link (lossless, unlimited), so an unconfigured channel
# behaves exactly like no realism at all.

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommsConfig:
    # Uniform (distance-independent) Bernoulli loss per (message, recipient).
    drop_prob: float = 0.0
    # Per-recipient delivered-payload budget per tick; None means unlimited.
    max_bytes_per_tick: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.drop_prob <= 1.0:
            raise ValueError(f"drop_prob must be in [0, 1], got {self.drop_prob}")
        if self.max_bytes_per_tick is not None and self.max_bytes_per_tick < 0:
            raise ValueError(
                f"max_bytes_per_tick must be >= 0, got {self.max_bytes_per_tick}"
            )
