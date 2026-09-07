"""Procedural map generation and the episode seed manifest.

Covers the generator invariants (determinism, seed=None, dense maps, enforced
min_free_fraction, descriptive exhaustion, requested vs realized density,
input validation) and the SeedSequence-based stream splitting in seeding.py.
"""

import numpy as np
import pytest

from map import Map, MapGenerationError
from seeding import STREAM_NAMES, derive_episode_seeds


# -- Map generation --------------------------------------------------------


def test_supplied_seed_is_deterministic():
    a = Map(width=20, height=20, seed=1234, obstacle_density=0.3)
    b = Map(width=20, height=20, seed=1234, obstacle_density=0.3)
    assert np.array_equal(a.grid, b.grid)
    # A different seed almost surely yields a different grid.
    c = Map(width=20, height=20, seed=9999, obstacle_density=0.3)
    assert not np.array_equal(a.grid, c.grid)


def test_seed_none_works():
    # The old generator did `None += 1` on retry; seed=None must just work.
    m = Map(width=16, height=16, seed=None, obstacle_density=0.3)
    assert m.grid.shape == (16, 16)
    assert m.vacancies > 0


def test_retry_does_not_mutate_configured_seed_or_attempts():
    # Retries run on a local derived stream, so the configured knobs are left
    # untouched (the old generator did `self.seed += 1` and decremented attempts).
    m = Map(width=16, height=16, seed=7, obstacle_density=0.4,
            min_free_fraction=0.3, max_generation_attempts=100)
    assert m.seed == 7
    assert m.max_generation_attempts == 100
    # Re-running with the same seed still reproduces the map.
    again = Map(width=16, height=16, seed=7, obstacle_density=0.4,
                min_free_fraction=0.3, max_generation_attempts=100)
    assert np.array_equal(m.grid, again.grid)


def test_dense_map_generates_without_crashing():
    # Dense + seed=None is the exact combination that used to crash. It must
    # either produce a valid map or raise MapGenerationError -- never TypeError.
    m = Map(width=24, height=24, seed=None, obstacle_density=0.6,
            min_free_fraction=0.1)
    assert m.vacancies > 0
    assert float(np.mean(m.grid == 0)) >= 0.1


def test_min_free_fraction_is_enforced():
    for seed in range(20):
        m = Map(width=18, height=18, seed=seed, obstacle_density=0.45,
                min_free_fraction=0.3)
        assert float(np.mean(m.grid == 0)) >= 0.3


def test_exhaustion_raises_descriptive_error():
    # An all-wall grid (density 1.0) can never meet any positive free fraction.
    with pytest.raises(MapGenerationError) as excinfo:
        Map(width=12, height=12, seed=0, obstacle_density=1.0,
            min_free_fraction=0.3, max_generation_attempts=5)
    msg = str(excinfo.value)
    assert "5 attempts" in msg
    assert "min_free_fraction" in msg


def test_requested_and_realized_density_are_distinct_fields():
    m = Map(width=20, height=20, seed=3, obstacle_density=0.3)
    assert m.requested_obstacle_density == 0.3
    # Border walls + component pruning raise the realized density above 0.3.
    assert m.realized_obstacle_density != m.requested_obstacle_density
    assert m.realized_obstacle_density == pytest.approx(float(np.mean(m.grid == 1)))


def test_derived_fields_always_agree():
    m = Map(width=15, height=21, seed=5, obstacle_density=0.3)
    assert m.grid.shape == (m.height, m.width) == (21, 15)
    assert m.vacancies == int(np.count_nonzero(m.grid == 0))
    assert len(m.free_cells) == m.vacancies


def test_free_space_is_connected():
    m = Map(width=20, height=20, seed=11, obstacle_density=0.4)
    free = {tuple(c) for c in m.free_cells}
    start = next(iter(free))
    seen, stack = {start}, [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (r + dr, c + dc)
            if nb in free and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    assert len(seen) == len(free)


def test_input_validation():
    with pytest.raises((ValueError, TypeError)):
        Map(width=0, height=10, seed=0)
    with pytest.raises((ValueError, TypeError)):
        Map(width=10, height=-1, seed=0)
    with pytest.raises(ValueError):
        Map(width=10, height=10, seed=0, obstacle_density=1.5)
    with pytest.raises(ValueError):
        Map(width=10, height=10, seed=0, min_free_fraction=2.0)
    with pytest.raises(ValueError):
        Map(width=10, height=10, seed=0, max_generation_attempts=0)


def test_named_map_loads_and_sets_dimensions():
    m = Map(width=10, height=10, name="spiral")
    from default_maps import DEFAULT_MAPS
    assert np.array_equal(m.grid, np.array(DEFAULT_MAPS["spiral"]["grid"], dtype=int))
    assert (m.height, m.width) == m.grid.shape
    assert m.vacancies == int(np.count_nonzero(m.grid == 0))


def test_unknown_named_map_raises():
    with pytest.raises(ValueError):
        Map(width=10, height=10, name="does_not_exist")


# -- seed manifest ---------------------------------------------------------


def test_manifest_is_deterministic_and_streams_distinct():
    a = derive_episode_seeds(42)
    b = derive_episode_seeds(42)
    assert a == b
    values = [getattr(a, name) for name in STREAM_NAMES]
    assert len(set(values)) == len(values)  # independent streams differ
    assert a.root == 42


def test_manifest_seed_none_is_replayable():
    a = derive_episode_seeds(None)
    # Re-deriving from the recorded root reproduces every stream.
    b = derive_episode_seeds(a.root)
    assert a == b


def test_overrides_hold_other_streams_fixed():
    base = derive_episode_seeds(7)
    varied = derive_episode_seeds(7, overrides={"spawn": 999})
    assert varied.spawn == 999
    # Every other stream is untouched -> "same map, different spawns".
    assert varied.map == base.map
    assert varied.comms == base.comms
    assert varied.policy == base.policy


def test_unknown_override_stream_raises():
    with pytest.raises(ValueError):
        derive_episode_seeds(7, overrides={"not_a_stream": 1})
