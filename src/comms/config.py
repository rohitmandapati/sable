# Configuration for the communication channel's transport realism
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

    # -- distance / range (stage 2) ------------------------------------------
    # Euclidean sender->recipient distance (in cells) at which the link fails:
    # this is both a hard cutoff (beyond it, always dropped) and the distance at
    # which path-loss reaches certain loss. None disables all distance effects.
    comm_range: float | None = None
    # Distance within which there is no path-loss penalty at all.
    reliable_range: float = 0.0
    # Path-loss curve exponent between reliable_range and comm_range (1 = linear,
    # >1 = reliable up close then a sharp cliff, <1 = degrades early).
    distance_falloff: float = 2.0

    # -- obstacle occlusion (stage 2) ----------------------------------------
    # Added drop probability per ground-truth wall cell crossed by the
    # sender->recipient line of sight. 0 disables occlusion; >= 1 makes a single
    # wall fully block the link.
    wall_attenuation: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.drop_prob <= 1.0:
            raise ValueError(f"drop_prob must be in [0, 1], got {self.drop_prob}")
        if self.max_bytes_per_tick is not None and self.max_bytes_per_tick < 0:
            raise ValueError(
                f"max_bytes_per_tick must be >= 0, got {self.max_bytes_per_tick}"
            )
        if self.comm_range is not None and self.comm_range <= 0.0:
            raise ValueError(f"comm_range must be > 0, got {self.comm_range}")
        if self.reliable_range < 0.0:
            raise ValueError(
                f"reliable_range must be >= 0, got {self.reliable_range}"
            )
        if self.comm_range is not None and self.reliable_range >= self.comm_range:
            raise ValueError(
                f"reliable_range ({self.reliable_range}) must be < comm_range "
                f"({self.comm_range})"
            )
        if self.distance_falloff <= 0.0:
            raise ValueError(
                f"distance_falloff must be > 0, got {self.distance_falloff}"
            )
        if self.wall_attenuation < 0.0:
            raise ValueError(
                f"wall_attenuation must be >= 0, got {self.wall_attenuation}"
            )
