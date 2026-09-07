from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from default_maps import DEFAULT_MAPS
from seeding import validate_attempts, validate_dimension, validate_fraction

class MapGenerationError(RuntimeError):
    """Raised when no random map meeting the invariants could be generated."""

@dataclass
class Map:
    width: int
    height: int
    seed: int | None = None
    obstacle_density: float = 0.5
    min_free_fraction: float = 0.3
    max_generation_attempts: int = 100
    name: str | None = None

    grid: np.ndarray = field(init=False, repr=False)
    free_cells: np.ndarray = field(init=False, repr=False)
    vacancies: int = field(init=False)
    requested_obstacle_density: float = field(init=False)
    realized_obstacle_density: float = field(init=False)

    def __post_init__(self) -> None:
        if self.name is not None:
            self._load_named(self.name)
        else:
            self._generate_random()
        self._finalize()

    # -- construction paths --------------------------------------------------

    def _load_named(self, name: str) -> None:
        entry = DEFAULT_MAPS.get(name)
        if entry is None:
            raise ValueError(
                f"Unknown default map {name!r}. Available: {sorted(DEFAULT_MAPS)}"
            )
        self.grid = np.array(entry["grid"], dtype=int)
        # The grid establishes the dimensions; a named map is fixed, so its
        # requested and realized densities are one and the same.
        self.height, self.width = (int(self.grid.shape[0]), int(self.grid.shape[1]))
        self.requested_obstacle_density = float(np.mean(self.grid == 1))

    def _generate_random(self) -> None:
        self.width = validate_dimension("width", self.width)
        self.height = validate_dimension("height", self.height)
        self.obstacle_density = validate_fraction("obstacle_density", self.obstacle_density)
        self.min_free_fraction = validate_fraction("min_free_fraction", self.min_free_fraction)
        self.max_generation_attempts = validate_attempts(
            "max_generation_attempts", self.max_generation_attempts
        )
        self.requested_obstacle_density = float(self.obstacle_density)

        # Each retry draws from an independent child stream of the configured
        # seed. This never mutates self.seed or an attempt counter, so the same
        # seed reproduces the same map, and seed=None still works (SeedSequence
        # draws fresh entropy). The attempt that first clears the free-fraction
        # invariant wins.
        root = np.random.SeedSequence(self.seed)
        for attempt_seq in root.spawn(self.max_generation_attempts):
            candidate = self._build_grid(np.random.default_rng(attempt_seq))
            if float(np.mean(candidate == 0)) >= self.min_free_fraction:
                self.grid = candidate
                return

        raise MapGenerationError(
            f"failed to generate a {self.width}x{self.height} map with "
            f"free fraction >= {self.min_free_fraction} at requested obstacle "
            f"density {self.requested_obstacle_density} within "
            f"{self.max_generation_attempts} attempts; lower the density or "
            f"min_free_fraction, or raise max_generation_attempts"
        )

    def _build_grid(self, rng: np.random.Generator) -> np.ndarray:
        # Random obstacles at the requested density, with a solid wall border,
        # then pruned to a single connected free region.
        grid = (rng.random((self.height, self.width)) < self.requested_obstacle_density).astype(int)
        grid[0, :] = 1
        grid[-1, :] = 1
        grid[:, 0] = 1
        grid[:, -1] = 1
        return self._keep_largest_component(grid)

    # -- finalisation & invariants ------------------------------------------

    def _finalize(self) -> None:
        # Recompute every derived field from the single source of truth (grid)
        # so free_cells / vacancies / dimensions / shape can never disagree.
        self.height, self.width = (int(self.grid.shape[0]), int(self.grid.shape[1]))
        self.free_cells = np.argwhere(self.grid == 0)
        self.vacancies = int(len(self.free_cells))
        self.realized_obstacle_density = float(np.mean(self.grid == 1))
        self._assert_invariants()

    def _assert_invariants(self) -> None:
        assert self.grid.shape == (self.height, self.width), "grid shape disagrees with dimensions"
        assert self.vacancies == int(np.count_nonzero(self.grid == 0)), "vacancies disagrees with grid"
        assert len(self.free_cells) == self.vacancies, "free_cells disagrees with vacancies"
        assert self.vacancies > 0, "map has no free cells"
        # Free space is a single connected component by construction.
        assert self._free_is_connected(self.grid), "free space is not connected"

    # -- connected-component helpers (pure; operate on a passed grid) --------

    @staticmethod
    def _keep_largest_component(grid: np.ndarray) -> np.ndarray:
        components = Map._free_components(grid)
        if not components:
            return np.ones_like(grid)
        best = max(components, key=len)
        out = np.ones_like(grid)
        for r, c in best:
            out[r, c] = 0
        return out

    @staticmethod
    def _free_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
        height, width = grid.shape
        seen = np.zeros_like(grid, dtype=bool)
        components: list[list[tuple[int, int]]] = []
        for sr in range(height):
            for sc in range(width):
                if grid[sr, sc] != 0 or seen[sr, sc]:
                    continue
                stack = [(sr, sc)]
                seen[sr, sc] = True
                cells: list[tuple[int, int]] = []
                while stack:
                    r, c = stack.pop()
                    cells.append((r, c))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = r + dr, c + dc
                        if (
                            0 <= nr < height
                            and 0 <= nc < width
                            and grid[nr, nc] == 0
                            and not seen[nr, nc]
                        ):
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                components.append(cells)
        return components

    @staticmethod
    def _free_is_connected(grid: np.ndarray) -> bool:
        components = Map._free_components(grid)
        return len(components) == 1
