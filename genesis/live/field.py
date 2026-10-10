"""The static flow field behind a live session.

Built once at startup from the recipe's field parameters (`scale`, `detail`,
`turns`, `swirl`), then cached for the life of the session. Rebuilding the
noise field per frame is the single most expensive thing the still renderer
does; the live engine never does it.

The step length is clamped with the same turn-per-step guard as the still
renderer, using the field's measured angular gradient. Cursor input paints
into a separately bounded stir layer in `fluid.py`, so the guard's guarantee
is preserved by construction rather than re-measured every frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .. import noise
from ..generators import flowfield as flowfield_module
from ..rng import stream


@dataclass(frozen=True)
class LiveField:
    """Cached field data for advection at full resolution."""

    width: int
    height: int
    cos_a: np.ndarray
    sin_a: np.ndarray
    gradient: float
    step_x: float
    step_y: float


def build_field(params: dict[str, Any], seed: int, width: int, height: int) -> LiveField:
    """Build and cache the static angle field for a live session.

    `params` carries the still-image field parameters; only the field-shaping
    keys are read (`scale`, `detail`, `turns`, `swirl`, `step_size`). Trail
    counts, exposure, and post settings do not apply here: the live engine
    advects a dense dye field and has a fixed display response.
    """
    aspect = noise.tileable_aspect(width, height)
    xs, ys = noise.coordinate_grid(width, height, aspect)
    angle = flowfield_module._angle_field(
        noise.permutation_table(stream(seed, "field")), xs, ys, params
    )
    gradient = flowfield_module.angular_gradient(angle)
    allowed = flowfield_module.max_step_for_gradient(gradient, width)
    step_size = min(float(params.get("step_size", 0.001)), allowed)
    angle32 = angle.astype(np.float32)
    return LiveField(
        width=width,
        height=height,
        cos_a=np.cos(angle32),
        sin_a=np.sin(angle32),
        gradient=gradient,
        step_x=step_size * width,
        step_y=step_size * height,
    )
