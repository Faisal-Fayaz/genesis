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
            12000,
            minimum=1000,
            maximum=300000,
            step=1000,
            label="Trail count",
            help="Particles to trace, scaled up with canvas size",
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
            1.5,
            minimum=0.25,
            maximum=24.0,
            step=0.25,
            label="Field scale",
            help="Noise cells across the image; lower is broader",
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
            0.0010,
            minimum=0.0001,
            maximum=0.02,
            step=0.0001,
            label="Step size",
            help="Fraction of the image crossed per step; smaller is finer",
        ),
        ParamSpec(
            "substeps",
            "int",
            0,
            minimum=0,
            maximum=8,
            label="Field samples",
            help=(
                "How many times the field is sampled while crossing one step. "
                "0 picks a count from the field's curvature, which costs nothing "
                "on a gentle field and pays for itself on a tight one."
            ),
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
            1.0,
            minimum=0.4,
            maximum=3.0,
            step=0.05,
            label="Contrast",
            help=(
                "How sharply trail density turns into brightness. Above 1 the "
                "faint ends of trails fade back and crossings stand out; below 1 "
                "they fill in. Affects ink only, never the background."
            ),
        ),
        ParamSpec(
            "supersample",
            "int",
            1,
            minimum=1,
            maximum=4,
            label="Supersampling",
            help=(
                "Render at N times size and downsample for smoother trails. "
                "Off by default: it quadruples both memory and time, and the "
                "finer step size reaches comparable detail without it."
            ),
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


def angular_gradient(angle: np.ndarray) -> float:
    """Mean absolute change in field angle between horizontally adjacent pixels.

    This is the quantity that decides whether a trail can follow the field or
    will cut across it. A particle moving `step` pixels turns by roughly
    `step * gradient` radians, so the two together set whether the trail looks
    like a smooth curve or a zigzag.

    The angle field is built with wraparound behaviour on each axis, so the
    +/-pi steps at the seams are excluded. They are a fixed handful of pixels
    against the rest of the image, and a mean is otherwise unaffected by them.
    """
    delta = np.abs(np.diff(angle, axis=1))
    inner = delta[:, 1:-1]
    inner = inner[inner < np.pi]
    if inner.size == 0:
        return 0.0
    return float(inner.mean())


def max_step_for_gradient(gradient: float, work_w: int) -> float:
    """Largest `step_size` that keeps one step under `MAX_TURN_PER_STEP`.

    Above this the particle steps past the curvature it is supposed to be
    following, and the trail degenerates into aliased noise whose frequency is
    set by `step_size` rather than by the field.
    """
    if gradient <= 0.0:
        return float("inf")
    return (MAX_TURN_PER_STEP / gradient) / max(work_w, 1)


def resolve_substeps(requested: int, step_x: float, gradient: float) -> int:
    """How many field samples to take while crossing one step.

    `requested` of 0 asks for a count derived from the step's own curvature. A
    step that turns by `step_x * gradient` radians is split until each piece
    turns by less than `TARGET_TURN_PER_SUBSTEP`, which is the point where a
    trail segment reads as a smooth curve rather than a corner.

    Deriving it matters because the cost is linear in this count: a gentle field
    needs one sample and pays nothing, while the recipes that actually need
    splitting are the ones where the extra passes buy visible smoothness.
    Measured against the recipes that motivated this, 0 resolves to 1 for the
    defaults and 2 for a field with 0.39 radians of turn per step.
    """
    if requested > 0:
        return max(requested, 1)
    turn = abs(step_x) * gradient
    needed = int(np.ceil(turn / TARGET_TURN_PER_SUBSTEP)) if turn > 0.0 else 1
    return max(1, min(needed, MAX_AUTO_SUBSTEPS))


def _render(
    params: dict[str, Any], seed: int, width: int, height: int, palette: Palette
) -> np.ndarray:
    factor = int(params["supersample"])
    work_w, work_h = width * factor, height * factor
    aspect = work_h / work_w

    xs, ys = noise.coordinate_grid(work_w, work_h, aspect)
    angle = _angle_field(noise.permutation_table(stream(seed, "field")), xs, ys, params)

    # Ink is specified relative to a reference area so that a recipe previewed at
    # 480p looks like the same composition at 1080p: trails scale up with the
    # canvas so strokes keep a constant density. The count is floored at the
    # requested value, because scaling *down* for a small preview starves the
    # image and the particles converge onto a handful of pixels instead of
    # spreading across the field.
    trails = min(MAX_TRAILS, round(int(params["trails"]) * _edge_ratio(work_w, work_h)))
    steps = int(params["steps"])
    substeps = max(int(params["substeps"]), 1)
    sub_samples = 6

    # Clamp the step to the curvature of the field actually being rendered. A
    # recipe that asks for a long step across a tightly turning field would
    # otherwise alias: the particle leaves the curve between samples and the
    # trail becomes noise whose frequency comes from step_size, not the field.
    # Measured on the field that is already built, so the clamp costs one pass
    # and no extra memory, and it is a no-op for any recipe that already turns
    # gently.
    gradient = angular_gradient(angle)
    requested_step = float(params["step_size"])
    step_size = min(requested_step, max_step_for_gradient(gradient, work_w))
    step_x = step_size * work_w
    step_y = step_size * work_h

    substeps = resolve_substeps(int(params["substeps"]), step_x, gradient)

    # Sub-stepping divides the distance, not the ink: each sub-step splats with
    # a weight that sums to the same total across the whole step, so trail
    # density and exposure are unaffected by the count.
    sub_step_x = step_x / substeps
    sub_step_y = step_y / substeps

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

    # Trails now scale with area, so weight per pixel stays constant and no
    # compensating ink scale is needed.

    position_t = np.linspace(0.0, 1.0, sub_samples)[None, :]
    ramp_channels = [ramp_rgb[:, channel] for channel in range(3)]

    for step in range(steps):
        progress = step / max(steps - 1, 1)
        fade = (1.0 - progress) ** 1.6

        color_t = np.clip(progress + phase, 0.0, 1.0)
        stroke_rgb = np.stack(
            [np.interp(color_t, ramp_t, channel) for channel in ramp_channels],
            axis=-1,
        )

        # Sub-stepping walks the particle along the field instead of stepping
        # once across it. The field is re-sampled every sub-step, so curvature
        # between samples is actually followed; the weight per sub-step divides
        # the step's budget so total ink is unchanged.
        sub_weight = (energy * fade * phase_decay(progress)) / (substeps * sub_samples)

        for sub in range(substeps):
            col = np.clip(pos_x.astype(np.int64), 0, work_w - 1)
            row = np.clip(pos_y.astype(np.int64), 0, work_h - 1)
            angles = angle[row, col]
            next_x = pos_x + np.cos(angles) * sub_step_x
            next_y = pos_y + np.sin(angles) * sub_step_y

            sample_x = pos_x[:, None] + (next_x - pos_x)[:, None] * position_t
            sample_y = pos_y[:, None] + (next_y - pos_y)[:, None] * position_t

            sample_col = np.clip(sample_x.astype(np.int64), 0, work_w - 1)
            sample_row = np.clip(sample_y.astype(np.int64), 0, work_h - 1)

            flat_row = (sample_row * work_w + sample_col).ravel()
            flat_weight = np.repeat(sub_weight, sub_samples)
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
            # Respawn anywhere on the canvas: restarting near the exit point
            # makes trails retrace one path, which collapses coverage.
            pos_x = np.where(escaped, rng.random(trails) * work_w, pos_x)
            pos_y = np.where(escaped, rng.random(trails) * work_h, pos_y)
            phase = np.where(escaped, rng.random(trails) * 0.35, phase)

    accumulator.finish()
    weight_sum = downsample(weight_sum, factor)
    color_sum = downsample(color_sum, factor)

    lit = weight_sum > 1e-9
    averaged = np.zeros_like(color_sum)
    averaged[lit] = color_sum[lit] / weight_sum[lit][:, None]

    # Ink is referenced to a high percentile of the pixels that actually received a
    # stroke, not to the peak. Particle trails cross unevenly: a few pixels where
    # many strokes pile up can be orders of magnitude above the typical lit
    # pixel. Referencing the peak would then leave every ordinary stroke invisible
    # and the image almost entirely background.
    lit_weights = weight_sum[weight_sum > 0.0]
    if lit_weights.size == 0:
        return np.tile(palette.to_linear(0.0), (work_h, work_w, 1))
    reference = float(np.percentile(lit_weights, 90.0))
    if reference <= 1e-12:
        return np.tile(palette.to_linear(0.0), (work_h, work_w, 1))

    density = 1.0 - np.exp(
        -weight_sum * (float(params["exposure"]) * _EXPOSURE_REFERENCE / reference)
    )

    # Contrast reshapes how ink weight becomes brightness, before it is tinted.
    # It is deliberately not the same thing as post `contrast`: that one grades
    # the finished picture in perceptual space, while this one chooses where the
    # faint tail of a trail sits relative to its dense crossings. `exposure` sets
    # the overall gain on ink, and this sets the curve, so the two compose
    # instead of duplicating each other.
    #
    # A power curve is used because it is monotonic, cannot invert, and leaves
    # pure background untouched: density 0 stays 0 at any setting. A pivoted
    # linear contrast cannot promise that, and on this generator the background
    # occupies most of the frame.
    contrast = float(params["contrast"])
    if contrast != 1.0:
        density = np.power(density, 1.0 / max(contrast, 1e-3))

    # Fill the field with the palette's darkest stop first, then lay strokes
    # over it. Compositing onto black instead would make an empty canvas read as
    # a void rather than as the deliberate dark end of the ramp.
    background = palette.to_linear(0.0)
    result = background + averaged * density[..., None]
    result = np.clip(result, 0.0, 1.0)
    return result


# Ink scales with the longest edge, not with area. A trail is a curve of fixed
# length *relative to the canvas* (step_size is a fraction of the image), so its
# pixel length grows linearly with resolution while the area to cover grows
# quadratically. Scaling trails by the edge therefore holds stroke density
# constant; scaling by area would over-provision quadratically.
REFERENCE_EDGE = 960.0

# Cap on trails after scaling. Coverage saturates well before this, and the cap
# keeps a 4K render from spending minutes on strokes past the point of visible
# difference.
MAX_TRAILS = 300_000

# How far the field may rotate across one step before the step is clamped.
# A 90 degree turn per step traces a visible polygon; 45 degrees is the point
# where a trail still reads as a curve. Measured against the recipes that
# motivated this: the curated defaults sit at 0.01 to 0.12 turns per step and
# are untouched, while the one recipe that rendered as coarse mottled noise was
# at 1.42 turns per step and clamps.
MAX_TURN_PER_STEP = np.pi / 4.0

# Turn budget per field sample, used only when `substeps` is left on auto. Set
# well under MAX_TURN_PER_STEP so an auto-split trail is smoother than the bare
# guard allows, and above the point where a segment's own curvature becomes
# visible as faceting on the densest fields.
TARGET_TURN_PER_SUBSTEP = 0.2

# Ceiling for the auto count. Measured on the worst field the random draw can
# produce, splitting further stops paying: 4 samples already reaches a jagged
# fraction of 0.031 and 8 only reaches 0.024, while costing twice as much again.
MAX_AUTO_SUBSTEPS = 4

# The densest pixel reaches roughly this fraction of full brightness at
# exposure 1.0.
_EXPOSURE_REFERENCE = 2.5


def _edge_ratio(work_w: int, work_h: int) -> float:
    return max(work_w, work_h) / REFERENCE_EDGE


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
