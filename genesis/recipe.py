"""Recipe: the complete, self-contained description of one image.

A recipe is the unit of reproducibility. It names a generator, carries that
generator's validated parameters, and states the size, palette, and post
processing. Given a recipe and the pinned library versions, the output is
byte-identical, which is what makes images diffable and regenerable forever.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import post as post_module
from . import registry
from .canon import canonical_json, derive_seed, stable_hash
from .color import Palette
from .rng import random_seed, stream, uniform

RECIPE_VERSION = 1
# Ramp order matters: stop zero is the background the renderer fills the canvas
# with, so every palette has to run dark to light. `ink` originally ran the
# other way and produced a near-white wallpaper, which left desktop icons
# sitting on a bright field. It is ordered like the others now, and a test
# asserts the property so a new palette cannot repeat the mistake.
PRESETS: dict[str, list[str]] = {
    "ember": ["#0d0a12", "#2b1b3d", "#7a2f5c", "#e0603f", "#ffd08a"],
    "tide": ["#04121f", "#0b3a53", "#1d7a8c", "#63c7b3", "#e9f6ef"],
    "moss": ["#0b1410", "#1c3324", "#3f6b45", "#8fbf6f", "#eaf0d8"],
    "ink": ["#0d1220", "#28324d", "#5b6b8c", "#b8c0cc", "#f2efe9"],
    "neon": ["#08030f", "#2d0a4a", "#7b1fa2", "#e91e8c", "#ffd166"],
    "slate": ["#101215", "#2b3138", "#5c6773", "#9aa7b4", "#e6ebef"],
    "solar": ["#1a0f05", "#4d2a0d", "#a85c15", "#f0a818", "#fff1c9"],
}


@dataclass
class Recipe:
    generator: str
    params: dict[str, Any] = field(default_factory=dict)
    width: int = 1920
    height: int = 1080
    palette_name: str = "tide"
    palette: list[str] | None = None
    seed: int | None = None
    theme: str = "dark"
    post: dict[str, Any] = field(default_factory=dict)
    version: int = RECIPE_VERSION

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> Recipe:
        # Loading a recipe has to work on its own. Without this, `Recipe.load`
        # only succeeds if something happened to import a generator first, which
        # leaves the library's stored recipes unreadable from a fresh process.
        registry.ensure_loaded()

        if self.version != RECIPE_VERSION:
            raise ValueError(
                f"recipe version {self.version} is not supported (expected {RECIPE_VERSION})"
            )
        if not isinstance(self.width, int) or self.width < 16 or self.width > 16384:
            raise ValueError(f"width must be an integer in 16..16384, got {self.width!r}")
        if not isinstance(self.height, int) or self.height < 16 or self.height > 16384:
            raise ValueError(f"height must be an integer in 16..16384, got {self.height!r}")
        if self.theme not in ("dark", "light"):
            raise ValueError(f"theme must be 'dark' or 'light', got {self.theme!r}")
        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, int)):
            raise ValueError(f"seed must be an integer or null, got {self.seed!r}")
        generator = registry.get(self.generator)
        self.params = generator.schema.validate(self.params)
        # Validate post now rather than at render time, so a bad recipe fails
        # where it is written instead of deep inside a render.
        post_module.resolve(self.post)
        if self.palette is not None and (
            not isinstance(self.palette, list) or not self.palette
        ):
            raise ValueError("palette must be a non-empty list of hex colors")
        return self

    @property
    def effective_seed(self) -> int:
        return derive_seed(self.to_dict())

    def effective_palette(self) -> Palette:
        colors = self.palette or PRESETS.get(self.palette_name) or PRESETS["tide"]
        return Palette(colors, self.palette_name)

    def resolved_post(self) -> dict[str, Any]:
        return post_module.resolve(self.post)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "generator": self.generator,
            "params": dict(self.params),
            "width": self.width,
            "height": self.height,
            "palette_name": self.palette_name,
            "palette": list(self.palette) if self.palette else None,
            "seed": self.seed,
            "theme": self.theme,
            "post": dict(self.post),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def canonical(self) -> str:
        return canonical_json(self.to_dict())

    def with_changes(self, **changes: Any) -> Recipe:
        merged = self.to_dict()
        merged.update(changes)
        return Recipe.from_dict(merged)

    def scaled(self, factor: float) -> Recipe:
        return self.with_changes(
            width=max(16, round(self.width * factor)),
            height=max(16, round(self.height * factor)),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Recipe:
        if not isinstance(data, dict):
            raise TypeError("a recipe must be a JSON object")
        known = {f for f in cls.__dataclass_fields__}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown recipe field(s) {unknown}; known fields are {sorted(known)}")
        return cls(
            generator=data["generator"],
            params=data.get("params", {}),
            width=data.get("width", 1920),
            height=data.get("height", 1080),
            palette_name=data.get("palette_name", "tide"),
            palette=data.get("palette"),
            seed=data.get("seed"),
            theme=data.get("theme", "dark"),
            post=data.get("post", {}),
            version=data.get("version", RECIPE_VERSION),
        )

    @classmethod
    def from_json(cls, text: str) -> Recipe:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"recipe is not valid JSON: {exc}") from exc
        return cls.from_dict(data)

    @classmethod
    def load(cls, path: str | Path) -> Recipe:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")
        return target

    def slug(self) -> str:
        seed_tag = format(self.effective_seed, "08x")
        return f"{self.generator}-{self.width}x{self.height}-{seed_tag}"

    def composition_key(self) -> str:
        """A size-independent identity for a composition.

        Same picture idea at any render size: generator, palette, theme, seed
        material, and parameters — but not dimensions. Live preference
        overrides are keyed by this so they survive display changes and the
        preview/full-size split.
        """
        seed_tag = format(self.effective_seed, "08x")
        params_tag = format(stable_hash(self.params) & ((1 << 48) - 1), "012x")
        return f"{self.generator}-{self.palette_name}-{self.theme}-{seed_tag}-{params_tag}"


def random_recipe(
    generator_name: str | None = None,
    rng: Any = None,
    **overrides: Any,
) -> Recipe:
    """Build a recipe with randomized parameters inside their valid ranges."""
    registry.ensure_loaded()
    if generator_name is None:
        if rng is None:
            raise ValueError("no generators are registered")
        choices = registry.names()
        generator_name = choices[int(rng.integers(len(choices)))]

    generator = registry.get(generator_name)
    seed_source = rng if rng is not None else stream(random_seed(stream(0, "entropy")), "params")
    params: dict[str, Any] = {}
    for spec in generator.schema:
        if spec.kind == "enum":
            params[spec.name] = str(
                spec.choices[int(seed_source.integers(len(spec.choices)))]
            )
        elif spec.kind == "bool":
            params[spec.name] = bool(seed_source.random() < 0.5)
        elif spec.kind in ("int", "float"):
            params[spec.name] = (
                spec.default if spec.name in DERIVED else _sample_number(spec, seed_source)
            )
        elif spec.kind == "color":
            params[spec.name] = _sample_color(spec, seed_source)
    palette_names = sorted(PRESETS)
    palette_name = palette_names[int(seed_source.integers(len(palette_names)))]
    fields: dict[str, Any] = {
        "generator": generator_name,
        "params": params,
        "palette_name": palette_name,
        "seed": None,
    }
    fields.update(overrides)
    return Recipe(**fields)


# Parameters whose cost multiplies with other parameters, so a random draw near
# the top of every range at once is not merely large but combinatorial. Kept
# adjacent to the sampling code so a new expensive parameter is accounted for
# here rather than discovered as a one-minute render.
# How far up its declared range a random draw may reach. A value of 1.0 allows
# the full range.
#
# These exist because the costs multiply. `trails` sets the length of a vectorized
# step and `steps` sets how many there are, so drawing both near their maxima
# costs the product: a random flowfield at supersample 2 took 52s at 1080p
# against 9s for the defaults. Every generator's render-time budget is roughly
# proportional to trails x steps x supersample^2, so each factor is capped well
# below its own maximum and the product lands in a sane range.
# How far a random draw may travel from one end of a parameter's range. A value
# of 1.0 allows the full range.
#
# `COSTLY` holds render-time multipliers down, because their costs multiply:
# `trails` sets the length of a vectorized step and `steps` sets how many there
# are, so drawing both near their maxima costs the product. A random flowfield
# at supersample 2 took 52s at 1080p against 9s for the defaults.
COSTLY = {
    "trails": 0.12,
    "steps": 0.25,
    "points": 0.35,
    "warmup": 0.15,
    "supersample": 0.0,  # never above the default, which is already correct
    "detail": 0.5,
}

# Parameters left at their default rather than drawn, because a random value is
# worse than no variation at all. Each one is either computed from other
# parameters or expresses a tradeoff where any fixed choice is arbitrary.
DERIVED = {"substeps"}

# Parameters that decide how bright or saturated the result looks. Their full
# ranges are there so a person can push past the defaults deliberately, but a
# random draw across the range produces a blown-out or muddy image rather than a
# variation of a good one: `hue_bias` at +0.55 more than doubled the mean
# luminance of a flowfield and washed out the dark end of the palette.
MOOD = {
    "hue_bias": 0.30,
    "exposure": 0.35,
    "contrast": 0.25,
    "saturation": 0.25,
    "gamma": 0.35,
    "zoom": 0.35,
    "brightness": 0.25,
    "lift": 0.2,
}

# Parameters that decide the *character* of a composition rather than its tone.
# Their ranges are wide because a person may want, say, a very broad field. A
# random draw across the whole range lands somewhere uninteresting instead of
# somewhere good: a flowfield at scale 13.8 with 7 octaves produced coarse
# mottled noise, where the tuned scale 1.5 gives layered silk. Held near the
# default, so "random" varies a composition rather than replacing it.
#
# `scale` and `detail` are held tighter than the rest because together they set
# how fast the field's angle changes per pixel, and that rate is what the
# generator's step guard is measured against. At these caps a draw reaches
# scale 3.0 and detail 5.7, which stays inside the guard's limit and inside the
# budget of one field sample. Looser caps still render correctly now, because
# the guard clamps the step, but they do so by shortening every trail, which
# changes the composition the draw was asking for rather than just its detail.
STRUCTURE = {
    "scale": 0.18,
    "turns": 0.3,
    "detail": 0.25,
    "step_size": 0.3,
    "steps": 0.25,
}


def _sample_number(spec: Any, rng: Any) -> float | int:
    """Draw a parameter value that is varied, affordable, and still looks good.

    Two caps apply. Expensive parameters (`COSTLY`) are held near the low end of
    their range, because their costs multiply. Parameters that control tone
    (`MOOD`) are held near their default, because the range was widened for
    deliberate experimentation and a random draw at its top produces a
    blown-out image. Everything else is free across its full range.
    """
    default = float(spec.default)
    low = float(spec.minimum) if spec.minimum is not None else default - 1.0
    high = float(spec.maximum) if spec.maximum is not None else default + 1.0

    if spec.name in COSTLY:
        fraction = COSTLY[spec.name]
        # Capping below the default is a no-op for parameters whose default is
        # already low, and holds the rest down to where a render stays quick.
        high = min(high, low + (high - low) * fraction)

    # Center the tone and structure caps on the default, so they constrain
    # deviation in both directions rather than only truncating upward.
    for table in (MOOD, STRUCTURE):
        if spec.name in table:
            span = (high - low) * table[spec.name]
            low = max(low, default - span)
            high = min(high, default + span)

    value = uniform(rng, low, high)
    value = 0.65 * value + 0.35 * default
    value = float(np.clip(value, min(low, high), max(low, high)))
    return round(value) if spec.kind == "int" else value


def _sample_color(spec: Any, rng: Any) -> str:
    """Draw a color near the declared default rather than anywhere in the cube.

    Fully random colors are usually muddy or too bright to read text over, which
    makes for a wallpaper nobody keeps.
    """
    base = str(spec.default).lstrip("#")
    channels = [
        int(base[index : index + 2], 16)
        for index in (0, 2, 4)
    ]
    nudged = [
        int(np.clip(channel + float(uniform(rng, -24, 24)), 0, 255))
        for channel in channels
    ]
    return "#" + "".join(f"{channel:02x}" for channel in nudged)
