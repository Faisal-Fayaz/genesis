"""Gradient noise built on seeded numpy operations.

A permutation table is drawn once from a named substream, then every lookup
reads from that table. Values are indexed by integer lattice coordinates, so the
same seed reproduces the same field on any machine.
"""

from __future__ import annotations

import numpy as np

_PERM_SIZE = 512


def permutation_table(rng: np.random.Generator) -> np.ndarray:
    table = rng.permutation(256).astype(np.int32)
    return np.concatenate([table, table])


def _fade(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    return a + t * (b - a)


def gradient_dot(hash_value: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Dot each gradient hash with its offset from the lattice point."""
    h = hash_value & 7
    u = np.where(h < 4, dx, dy)
    v = np.where(h < 4, dy, dx)
    return np.where((h & 1) == 0, u, -u) + np.where((h & 2) == 0, v, -v)


def perlin_2d(
    perm: np.ndarray, xs: np.ndarray, ys: np.ndarray, frequency: float = 1.0
) -> np.ndarray:
    """Perlin noise sampled at arbitrary coordinates.

    Coordinates are taken modulo the lattice, so the field repeats every
    `1 / frequency` units. That repetition is what lets wallpapers tile.
    """
    px = xs * frequency
    py = ys * frequency
    x0 = np.floor(px).astype(np.int64)
    y0 = np.floor(py).astype(np.int64)
    fx = px - x0
    fy = py - y0
    xi = (x0 & 255).astype(np.int32)
    yi = (y0 & 255).astype(np.int32)

    aa = perm[perm[xi] + yi]
    ab = perm[perm[xi] + yi + 1]
    ba = perm[perm[xi + 1] + yi]
    bb = perm[perm[xi + 1] + yi + 1]

    u = _fade(fx)
    v = _fade(fy)
    lower = _lerp(gradient_dot(aa, fx, fy), gradient_dot(ba, fx - 1.0, fy), u)
    upper = _lerp(gradient_dot(ab, fx, fy - 1.0), gradient_dot(bb, fx - 1.0, fy - 1.0), u)
    return _lerp(lower, upper, v)


def fbm(
    perm: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    octaves: int = 5,
    frequency: float = 1.0,
    lacunarity: float = 2.0,
    gain: float = 0.5,
) -> np.ndarray:
    """Fractal Brownian motion: octaves of Perlin noise at doubling frequency."""
    if octaves < 1:
        raise ValueError("octaves must be at least 1")
    total = np.zeros_like(xs, dtype=np.float64)
    amplitude = 1.0
    normalization = 0.0
    current = frequency
    for _ in range(octaves):
        total += amplitude * perlin_2d(perm, xs, ys, current)
        normalization += amplitude
        amplitude *= gain
        current *= lacunarity
    return total / normalization


def ridged(
    perm: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    octaves: int = 5,
    frequency: float = 1.0,
    lacunarity: float = 2.0,
    gain: float = 0.5,
) -> np.ndarray:
    """Ridged multifractal: folds the noise to make sharp crests, 0..1."""
    total = np.zeros_like(xs, dtype=np.float64)
    amplitude = 1.0
    normalization = 0.0
    current = frequency
    for _ in range(octaves):
        folded = 1.0 - np.abs(perlin_2d(perm, xs, ys, current))
        total += amplitude * folded * folded
        normalization += amplitude
        amplitude *= gain
        current *= lacunarity
    clipped = total / normalization
    return np.clip(clipped, 0.0, 1.0)


def coordinate_grid(width: int, height: int, aspect: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """A unit-square coordinate mesh, stretched to the pixel aspect ratio."""
    xs = np.linspace(0.0, 1.0, width, dtype=np.float64)
    ys = np.linspace(0.0, 1.0 * aspect, height, dtype=np.float64)
    return np.meshgrid(xs, ys)


def worley(
    rng: np.random.Generator, width: int, height: int, count: int = 64, jitter: float = 0.85
) -> np.ndarray:
    """Worley/cellular noise: distance to the nearest jittered feature point.

    Feature points are placed on a wrapped grid so the result tiles the same way
    the Perlin field does.
    """
    if count < 1:
        raise ValueError("count must be at least 1")
    side = int(np.ceil(np.sqrt(count)))
    positions = rng.random((side * side, 2))
    positions = (positions - 0.5) * jitter + 0.5
    positions = positions % 1.0

    xs = np.linspace(0.0, 1.0, width, dtype=np.float64)
    ys = np.linspace(0.0, 1.0, height, dtype=np.float64)
    gx, gy = np.meshgrid(xs, ys)

    nearest = np.full((height, width), np.inf)
    for px, py in positions:
        dx = np.abs(gx - px)
        dy = np.abs(gy - py)
        dx = np.minimum(dx, 1.0 - dx)
        dy = np.minimum(dy, 1.0 - dy)
        np.minimum(nearest, np.sqrt(dx * dx + dy * dy), out=nearest)
    return nearest


def normalize(field: np.ndarray) -> np.ndarray:
    """Rescale a field to 0..1 without collapsing on constant input."""
    low = float(np.min(field))
    high = float(np.max(field))
    if high - low < 1e-12:
        return np.zeros_like(field)
    return (field - low) / (high - low)
