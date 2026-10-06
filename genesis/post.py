"""Post processing.

These steps run after a generator has produced linear-light RGB. Each one is
declared in the recipe, so a post-processing change is a diffable edit to the
recipe rather than a code change. Every step reads from its own named substream
of the recipe seed, which keeps noise reproducible.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .params import ParamSpec, build_schema
from .rng import stream

SCHEMA = build_schema(
    [
        ParamSpec(
            "grain",
            "float",
            0.0,
            minimum=0.0,
            maximum=0.4,
            step=0.005,
            label="Grain",
            help="Filmic noise; sells the illusion of a physical print",
        ),
        ParamSpec(
            "grain_size",
            "float",
            1.0,
            minimum=0.5,
            maximum=4.0,
            step=0.1,
            label="Grain size",
            help="Pixel size of the grain clumps",
        ),
        ParamSpec(
            "vignette",
            "float",
            0.0,
            minimum=0.0,
            maximum=1.0,
            step=0.01,
            label="Vignette",
            help="Darkens the corners",
        ),
        ParamSpec(
            "bloom",
            "float",
            0.0,
            minimum=0.0,
            maximum=1.0,
            step=0.01,
            label="Bloom",
            help="Glows bright areas outward",
        ),
        ParamSpec(
            "bloom_radius",
            "float",
            24.0,
            minimum=2.0,
            maximum=160.0,
            step=1.0,
            label="Bloom radius",
        ),
        ParamSpec(
            "saturation",
            "float",
            1.0,
            minimum=0.0,
            maximum=2.0,
            step=0.02,
            label="Saturation",
        ),
        ParamSpec(
            "brightness",
            "float",
            1.0,
            minimum=0.2,
            maximum=2.5,
            step=0.02,
            label="Brightness",
        ),
        ParamSpec(
            "contrast",
            "float",
            1.0,
            minimum=0.4,
            maximum=2.5,
            step=0.02,
            label="Contrast",
        ),
        ParamSpec(
            "lift",
            "float",
            0.0,
            minimum=-0.15,
            maximum=0.25,
            step=0.005,
            label="Lift",
            help="Raises the black point, giving a faded matte look",
        ),
    ]
)


def resolve(overrides: dict[str, Any]) -> dict[str, Any]:
    """Merge a recipe's post settings over the defaults and validate them."""
    if not isinstance(overrides, dict):
        raise TypeError("post settings must be an object")
    unknown = sorted(set(overrides) - {spec.name for spec in SCHEMA})
    if unknown:
        raise ValueError(f"unknown post option(s) {unknown}")
    return SCHEMA.validate(overrides)


def is_identity(settings: dict[str, Any]) -> bool:
    defaults = SCHEMA.defaults()
    return all(abs(settings[key] - defaults[key]) < 1e-12 for key in settings if isinstance(settings[key], (int, float)))


def _luma(image: np.ndarray) -> np.ndarray:
    return image @ np.array([0.2126, 0.7152, 0.0722], dtype=image.dtype)


def apply(image: np.ndarray, seed: int, settings: dict[str, Any]) -> np.ndarray:
    """Run the full post chain over a linear-light image."""
    result = np.array(image, dtype=np.float64, copy=True)
    height, width = result.shape[:2]

    brightness = float(settings["brightness"])
    contrast = float(settings["contrast"])
    saturation = float(settings["saturation"])
    lift = float(settings["lift"])

    if brightness != 1.0 or contrast != 1.0 or saturation != 1.0 or lift != 0.0:
        result = (result - 0.5) * contrast + 0.5
        result = result * brightness + lift
        if saturation != 1.0:
            gray = _luma(result)[..., None]
            result = gray + (result - gray) * saturation
        result = np.clip(result, 0.0, 1.0)

    bloom = float(settings["bloom"])
    if bloom > 0.0:
        radius = max(int(settings["bloom_radius"]), 1)
        radius = min(radius, max(min(height, width) // 2, 1))
        result = result + bloom * _blur(_highpass(result), radius)

    vignette = float(settings["vignette"])
    if vignette > 0.0:
        ys = np.linspace(-1.0, 1.0, height, dtype=np.float64)[:, None]
        xs = np.linspace(-1.0, 1.0, width, dtype=np.float64)[None, :]
        falloff = np.sqrt((xs / (width / max(height, 1))) ** 2 + ys**2)
        falloff = falloff / max(float(falloff.max()), 1e-9)
        result = result * (1.0 - vignette * falloff**2)[..., None]

    grain = float(settings["grain"])
    if grain > 0.0:
        result = _grain(result, seed, grain, float(settings["grain_size"]))

    return np.clip(result, 0.0, 1.0)


def _highpass(image: np.ndarray) -> np.ndarray:
    """Isolate what should bloom: bright detail, not the whole image."""
    blurred = _blur(image, max(min(image.shape[:2]) // 12, 1))
    return np.clip(image - blurred, 0.0, None)


def _blur(image: np.ndarray, radius: int) -> np.ndarray:
    """Separable box blur, repeated to approximate a gaussian.

    Three passes of a box are close enough to a gaussian for a glow, and far
    cheaper than a true convolution at these radii.
    """
    result = image
    for axis in (0, 1):
        for _ in range(3):
            result = _box_axis(result, radius, axis)
    return result


def _box_axis(image: np.ndarray, radius: int, axis: int) -> np.ndarray:
    if radius < 1:
        return image
    width = 2 * radius + 1
    kernel = np.ones(width, dtype=np.float64) / width
    padding = [(0, 0)] * image.ndim
    padding[axis] = (radius, radius)
    padded = np.pad(image, padding, mode="edge")
    moved = np.apply_along_axis(
        lambda line: np.convolve(line, kernel, mode="valid"), axis, padded
    )
    return moved


def _grain(image: np.ndarray, seed: int, amount: float, size: float) -> np.ndarray:
    """Add luminance-weighted noise.

    Grain is strongest in the midtones and weaker in highlights, which is how
    film behaves and what keeps bright areas from looking speckled.
    """
    height, width = image.shape[:2]
    rng = stream(seed, "grain")
    block = max(round(size), 1)
    coarse_h = max(height // block, 1)
    coarse_w = max(width // block, 1)
    noise_field = rng.normal(0.0, 1.0, (coarse_h, coarse_w))
    if block > 1:
        row_index = (np.arange(height) / block).astype(np.int64).clip(0, coarse_h - 1)
        col_index = (np.arange(width) / block).astype(np.int64).clip(0, coarse_w - 1)
        noise_field = noise_field[row_index][:, col_index]
    luma = np.clip(_luma(image), 0.0, 1.0)
    weight = (1.0 - np.abs(luma - 0.5) * 2.0) ** 0.5
    return image + (noise_field * weight * amount)[..., None]
