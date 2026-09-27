"""Synthetic two-dimensional point clouds used as graph input for this example."""

from __future__ import annotations

import numpy as np

__all__ = ["MAX_POINTS", "MIN_POINTS", "PRESETS", "make_points"]

#: Synthetic 2D distributions the graph can be built on.
PRESETS: tuple[str, ...] = ("blobs", "moons", "circles", "spiral", "grid", "uniform")

#: Smallest and largest cloud the example accepts. The upper bound is not a memory limit but a runtime
#: one: the theoretical reference graphs in `theory.py` compare the DEG against the Delaunay graph, the
#: RNG and the MRNG, and their cost grows faster than the point count.
MIN_POINTS = 10
MAX_POINTS = 1200


def _blobs(num_points: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Isotropic Gaussian clusters with per-cluster spread."""
    clusters = int(np.clip(num_points // 200, 3, 8))
    centers = rng.uniform(0.0, 10.0, size=(clusters, 2))
    spread = rng.uniform(0.25, 0.75, size=(clusters, 1))
    group = rng.integers(0, clusters, size=num_points)
    return centers[group] + rng.standard_normal((num_points, 2)) * spread[group], group


def _moons(num_points: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Two interleaved half-moons, the classic non-linearly-separable toy set."""
    group = rng.integers(0, 2, size=num_points)
    theta = rng.uniform(0.0, np.pi, size=num_points)
    x = np.where(group == 0, np.cos(theta), 1.0 - np.cos(theta)) * 4.0
    y = np.where(group == 0, np.sin(theta), 0.5 - np.sin(theta)) * 4.0
    points = np.column_stack([x, y]) + rng.standard_normal((num_points, 2)) * 0.22
    return points, group


def _circles(num_points: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Concentric rings with radial noise."""
    rings = np.array([1.2, 2.6, 4.2])
    group = rng.integers(0, rings.size, size=num_points)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=num_points)
    radius = rings[group] + rng.standard_normal(num_points) * 0.09
    return np.column_stack([radius * np.cos(theta), radius * np.sin(theta)]), group


def _spiral(num_points: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Multiple spiral arms wound around the origin."""
    arms = 3
    group = np.arange(num_points) % arms
    radius = np.sqrt(rng.uniform(0.0, 1.0, size=num_points)) * 4.5
    theta = radius * 1.35 + group * (2.0 * np.pi / arms)
    points = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    return points + rng.standard_normal((num_points, 2)) * 0.07, group


def _grid(num_points: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Jittered square lattice, grouped by cell parity."""
    side = int(np.ceil(np.sqrt(num_points)))
    step = 10.0 / side
    cell = np.arange(num_points)
    column, row = cell % side, cell // side
    group = (column + row) % 2
    points = np.column_stack([column, row]).astype(np.float64) * step
    return points + rng.standard_normal((num_points, 2)) * (step * 0.22), group


def _uniform(num_points: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Homogeneous fill of the square — the structureless baseline."""
    return rng.uniform(0.0, 10.0, size=(num_points, 2)), np.zeros(num_points, dtype=np.int64)


_GENERATORS = {
    "blobs": _blobs,
    "moons": _moons,
    "circles": _circles,
    "spiral": _spiral,
    "grid": _grid,
    "uniform": _uniform,
}


def make_points(preset: str, num_points: int, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates a synthetic 2D point cloud.

    Returns `(points, groups)` where `points` is a `[n, 2]` float32 array — the DEG feature
    vectors and the drawing coordinates at once — and `groups` is a `[n]` int8 array holding
    the generating cluster, used only for coloring.

    Every cloud is shifted so the middle of its bounding box sits at the origin, which makes the
    coordinates straddle zero. Distances are untouched by a translation, so the graph is the same.
    """
    if preset not in _GENERATORS:
        raise ValueError(f"Unknown preset '{preset}'. Available: {', '.join(PRESETS)}")
    if not MIN_POINTS <= num_points <= MAX_POINTS:
        raise ValueError(f"A preset needs between {MIN_POINTS} and {MAX_POINTS} points, got {num_points}")

    points, group = _GENERATORS[preset](int(num_points), np.random.default_rng(seed))
    points = points - (points.min(axis=0) + points.max(axis=0)) / 2.0
    points = np.ascontiguousarray(points, dtype=np.float32)
    if not np.isfinite(points).all():
        raise AssertionError(f"Preset '{preset}' produced non-finite coordinates")
    return points, np.ascontiguousarray(group, dtype=np.int8)
