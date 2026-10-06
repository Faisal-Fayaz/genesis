"""Flow field: particles advected through a noise-driven vector field.

Each particle leaves a fading trail. Trails accumulate into a color buffer and
a density buffer; the color buffer is a weighted average of the strokes that
landed on each pixel, and the density drives a saturation curve. Doing both at
once avoids the flat clipping that additive-only splats produce, and keeps the
whole loop vectorized across particles.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import noise
from ..color import Palette
from ..params import ParamSpec, build_schema
from ..registry import register
from ..rng import stream
from ._common import SplatAccumulator, downsample

SCHEMA = build_schema(
    [
        ParamSpec(
            "trails",
            "int",
            4000,
            minimum=100,
            maximum=60000,
            step=100,
            label="Trail count",
            help="How many particles to trace",
        ),
        ParamSpec(
            "steps",
            "int",
            220,
            minimum=20,
            maximum=2000,
            step=10,
            label="Trail length",
            help="Steps each particle takes before it respawns",
        ),
        ParamSpec(
            "scale",
            "float",
            3.0,
            minimum=0.25,
            maximum=24.0,
            step=0.25,
            label="Field scale",
            help="Frequency of the noise field; lower is broader",
        ),
        ParamSpec(
            "detail",
            "int",
            5,
            minimum=1,
            maximum=9,
            label="Detail",
            help="fBm octaves; higher is finer",
        ),
        ParamSpec(
            "turns",
            "float",
            3.0,
            minimum=0.5,
            maximum=12.0,
            step=0.5,
            label="Turns",
            help="How many full rotations the field makes across the image",
        ),
        ParamSpec(
            "swirl",
            "bool",
            True,
            label="Swirl",
            help="Offset the field by 90 degrees to favor vortices",
        ),
        ParamSpec(
            "step_size",
            "float",
            0.0022,
            minimum=0.0002,
            maximum=0.02,
            step=0.0002,
            label="Step size",
            help="Fraction of the image crossed per step",
        ),
        ParamSpec(
            "color_count",
            "int",
            5,
            minimum=2,
            maximum=12,
            label="Palette stops",
            help="How many colors to sample from the palette",
        ),
        ParamSpec(
            "hue_bias",
            "float",
            0.0,
            minimum=-1.0,
            maximum=1.0,
            step=0.05,
            label="Hue bias",
            help="Shifts which part of the palette trails land on",
        ),
        ParamSpec(
            "exposure",
            "float",
            1.6,
            minimum=0.1,
            maximum=8.0,
            step=0.1,
            label="Exposure",
            help="How quickly overlapping trails saturate",
        ),
        ParamSpec(
            "contrast",
            "float",
            1.2,
            minimum=0.4,
            maximum=3.0,
            step=0.05,
            label="Contrast",
        ),
        ParamSpec(
            "supersample",
            "int",
            2,
            minimum=1,
            maximum=4,
            label="Supersampling",
            help="Render at N times size and downsample for smoother trails",
        ),
    ]
)


def _angle_field(
    perm: np.ndarray, xs: np.ndarray, ys: np.ndarray, params: dict[str, Any]
) -> np.ndarray:
    raw = noise.fbm(
        perm,
        xs,
        ys,
        octaves=int(params["detail"]),
        frequency=float(params["scale"]),
    )
    span = max(float(raw.max() - raw.min()), 1e-12)
    field = (raw - raw.min()) / span
    turns = float(params["turns"])
    angle = field * (2.0 * np.pi * turns)
    if params["swirl"]:
        angle = angle + np.pi / 2.0
    return angle


def _render(
    params: dict[str, Any], seed: int, width: int, height: int, palette: Palette
) -> np.ndarray:
    factor = int(params["supersample"])
    work_w, work_h = width * factor, height * factor
    aspect = work_h / work_w

    xs, ys = noise.coordinate_grid(work_w, work_h, aspect)
    angle = _angle_field(noise.permutation_table(stream(seed, "field")), xs, ys, params)

    trails = int(params["trails"])
    steps = int(params["steps"])
    step_x = float(params["step_size"]) * work_w
    step_y = float(params["step_size"]) * work_h
    sub_samples = 6

    # float32 halves the memory traffic of the splat buffers, which dominates
    # runtime at 1080p with supersampling on.
    color_sum = np.zeros((work_h, work_w, 3), dtype=np.float32)
    weight_sum = np.zeros((work_h, work_w), dtype=np.float32)

    rng = stream(seed, "particles")
    pos_x = rng.random(trails) * work_w
    pos_y = rng.random(trails) * work_h
    energy = rng.uniform(0.4, 1.0, trails)
    phase = rng.random(trails) * 0.35

    stops = max(int(params["color_count"]), 2)
    bias = float(params["hue_bias"])
    ramp_t = np.linspace(0.0, 1.0, stops)
    ramp_rgb = palette.to_linear(np.clip(ramp_t + bias, 0.0, 1.0))

    accumulator = SplatAccumulator(color_sum, weight_sum)

    # Total ink is fixed by trail/step counts, so ink per pixel falls as area
    # grows. Scaling weight by area keeps a composition's exposure identical at
    # preview and 1080p, which is what lets parameters be tuned cheaply.
    area_scale = (work_w * work_h) / REFERENCE_AREA

    for step in range(steps):
        col = np.clip(pos_x.astype(np.int64), 0, work_w - 1)
        row = np.clip(pos_y.astype(np.int64), 0, work_h - 1)
        angles = angle[row, col]
        next_x = pos_x + np.cos(angles) * step_x
        next_y = pos_y + np.sin(angles) * step_y

        progress = step / max(steps - 1, 1)
        fade = (1.0 - progress) ** 1.6
        weights = (energy * fade * phase_decay(progress) * area_scale) / sub_samples

        position_t = np.linspace(0.0, 1.0, sub_samples)[None, :]
        sample_x = pos_x[:, None] + (next_x - pos_x)[:, None] * position_t
        sample_y = pos_y[:, None] + (next_y - pos_y)[:, None] * position_t

        sample_col = np.clip(sample_x.astype(np.int64), 0, work_w - 1)
        sample_row = np.clip(sample_y.astype(np.int64), 0, work_h - 1)

        color_t = np.clip(progress + phase, 0.0, 1.0)
        stroke_rgb = np.stack(
            [np.interp(color_t, ramp_t, ramp_rgb[:, channel]) for channel in range(3)],
            axis=-1,
        )

        flat_row = (sample_row * work_w + sample_col).ravel()
        flat_weight = np.repeat(weights, sub_samples)
        flat_rgb = np.repeat(stroke_rgb, sub_samples, axis=0)

        accumulator.add(
            flat_row,
            flat_weight,
            flat_rgb * flat_weight[:, None],
        )

        pos_x, pos_y = next_x, next_y
        escaped = (pos_x < 0) | (pos_x >= work_w) | (pos_y < 0) | (pos_y >= work_h)
        if escaped.any():
            # np.where evaluates both branches, so draws stay full-size.
            pos_x = np.where(escaped, rng.random(trails) * work_w, pos_x)
            pos_y = np.where(escaped, rng.random(trails) * work_h, pos_y)
            phase = np.where(escaped, rng.random(trails) * 0.35, phase)

    accumulator.finish()
    weight_sum = downsample(weight_sum, factor)
    color_sum = downsample(color_sum, factor)

    lit = weight_sum > 1e-9
    averaged = np.zeros_like(color_sum)
    averaged[lit] = color_sum[lit] / weight_sum[lit][:, None]

    exposure = float(params["exposure"])
    density = 1.0 - np.exp(-weight_sum * exposure)
    result = averaged * density[..., None]

    background = palette.to_linear(0.0)
    result = background * (1.0 - density[..., None]) + result
    contrast = float(params["contrast"])
    result = np.clip((result - 0.5) * contrast + 0.5, 0.0, 1.0)
    return result


REFERENCE_AREA = 1_920.0 * 1_080.0


def phase_decay(progress: float) -> float:
    """Slow fade across a trail's whole life, independent of the fade curve."""
    return 0.35 + 0.65 * (1.0 - progress)


def render(
    params: dict[str, Any], seed: int, width: int, height: int, palette: Palette
) -> np.ndarray:
    return _render(params, seed, width, height, palette)


register(
    "flowfield",
    "Particles drifting through a noise vector field, colored by trail length",
    SCHEMA,
    render,
    family="vector",
)
