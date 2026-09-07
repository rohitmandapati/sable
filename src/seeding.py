# Deterministic per-subsystem seeding for one episode.
#
# An episode needs several *independent* streams of randomness (map generation,
# spawn placement, environment dynamics, communication, policy behaviour). If
# they shared one generator, changing (say) how many random draws the map costs
# would silently perturb spawns and comms. We derive each stream from a single
# root episode seed with numpy's SeedSequence, which is designed exactly for
# splitting one seed into statistically-independent child streams.


from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

# The independent streams every episode splits into. Order is fixed so a root
# seed maps to the same child streams across runs and versions.
STREAM_NAMES: tuple[str, ...] = ("map", "spawn", "dynamics", "comms", "policy")


@dataclass(frozen=True)
class EpisodeSeeds:
    root: int
    map: int
    spawn: int
    dynamics: int
    comms: int
    policy: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def derive_episode_seeds(
    root: int | None = None,
    *,
    overrides: dict[str, int] | None = None,
) -> EpisodeSeeds:
    seq = np.random.SeedSequence(root)
    # SeedSequence.entropy is the int we passed, or the freshly-drawn entropy
    # when root was None. Recording it makes a seed=None episode reproducible.
    root_value = int(seq.entropy)

    children = seq.spawn(len(STREAM_NAMES))
    values = {
        name: int(child.generate_state(1, dtype=np.uint32)[0])
        for name, child in zip(STREAM_NAMES, children)
    }

    if overrides:
        unknown = set(overrides) - set(STREAM_NAMES)
        if unknown:
            raise ValueError(
                f"unknown seed stream(s) {sorted(unknown)}; "
                f"valid streams: {list(STREAM_NAMES)}"
            )
        values.update({name: int(value) for name, value in overrides.items()})

    return EpisodeSeeds(root=root_value, **values)


# -- input validation shared by Map and Environment ------------------------


def validate_dimension(name: str, value: int) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def validate_fraction(name: str, value: float) -> float:
    if not isinstance(value, (int, float, np.floating, np.integer)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


def validate_attempts(name: str, value: int) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value
