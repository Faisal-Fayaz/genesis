"""Shared helpers for generators.

Generators return a float RGB array in linear light. Keeping them out of Pillow
means the whole render pipeline is pure numpy, which is both faster and easier
to reason about; conversion to a final image happens once, at the end.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from ..color import linear_to_srgb


def canvas(width: int, height: int, value: float = 0.0) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.float64)


def to_image(linear: np.ndarray) -> Image.Image:
    """Convert a linear-light float array to an 8-bit sRGB image."""
    srgb = np.clip(linear_to_srgb(linear), 0.0, 1.0)
    # Mode is inferred from the uint8 HxWx3 shape; Pillow 13 drops `mode=`.
    return Image.fromarray(np.round(srgb * 255.0).astype(np.uint8))


def alpha_composite_soft(
    buffer: np.ndarray, rgb: np.ndarray, alpha: np.ndarray
) -> np.ndarray:
    """Accumulate light additively while still respecting coverage.

    Particle splats want to saturate toward white where strokes overlap rather
    than the flat clipping that plain addition gives, so this lifts toward the
    light target as alpha approaches 1.
    """
    weight = np.clip(alpha, 0.0, 1.0)[..., None]
    return buffer + (rgb - buffer) * weight


def scatter_add(
    buffer: np.ndarray, flat_indices: np.ndarray, values: np.ndarray
) -> None:
    """Accumulate `values` into a flattened buffer at the given pixel indices.

    `np.bincount` is several times faster than `np.add.at` here because it
    accumulates in compiled C rather than through fancy-indexing assignment.
    The buffer is updated in place and keeps its shape.
    """
    size = buffer.size
    flat = buffer.reshape(size, -1)
    # A 2D buffer is a single-channel map; only 3D+ carries a channel axis.
    channels = 1 if buffer.ndim <= 2 else flat.shape[1]
    for channel in range(channels):
        weights = values if values.ndim == 1 else values[:, channel]
        flat[:, channel] += np.bincount(flat_indices, weights=weights, minlength=size)


class SplatAccumulator:
    """Buffers particle splats and folds them into buffers in batches.

    Each splat is a pixel index plus a color and a weight. Accumulating them one
    at a time means one pass over the whole buffer per step, which dominates
    runtime for long trails. Collecting a chunk of steps first turns hundreds of
    full-buffer passes into a handful, at the cost of holding the chunk in
    memory.
    """

    def __init__(
        self,
        color_buffer: np.ndarray,
        weight_buffer: np.ndarray,
        max_pending: int = 4_000_000,
    ) -> None:
        self._color = color_buffer
        self._weight = weight_buffer
        self._max_pending = max(int(max_pending), 1)
        self._indices: list[np.ndarray] = []
        self._weights: list[np.ndarray] = []
        self._colors: list[np.ndarray] = []
        self._pending = 0

    def add(self, flat_indices: np.ndarray, weights: np.ndarray, colors: np.ndarray) -> None:
        self._indices.append(np.asarray(flat_indices, dtype=np.int64))
        self._weights.append(np.asarray(weights, dtype=np.float64))
        self._colors.append(np.asarray(colors, dtype=np.float64))
        self._pending += flat_indices.size
        if self._pending >= self._max_pending:
            self.flush()

    def flush(self) -> None:
        if not self._indices:
            return
        indices = np.concatenate(self._indices)
        weights = np.concatenate(self._weights)
        colors = np.concatenate(self._colors)
        self._indices.clear()
        self._weights.clear()
        self._colors.clear()
        self._pending = 0
        scatter_add(self._weight, indices, weights)
        scatter_add(self._color, indices, colors)

    def finish(self) -> tuple[np.ndarray, np.ndarray]:
        self.flush()
        return self._color, self._weight


def supersample_factor(params: dict[str, Any], default: int = 2) -> int:
    return max(1, min(4, int(params.get("supersample", default))))


def downsample(array: np.ndarray, factor: int) -> np.ndarray:
    """Box-average by `factor`, preserving the trailing axis structure.

    The trailing axis is named explicitly rather than left as -1, so a 2D
    buffer stays 2D instead of silently gaining a length-1 channel.
    """
    if factor <= 1:
        return array
    height, width = array.shape[0] // factor, array.shape[1] // factor
    trimmed = array[: height * factor, : width * factor]
    block_shape = (height, factor, width, factor) + array.shape[2:]
    return trimmed.reshape(block_shape).mean(axis=(1, 3))
