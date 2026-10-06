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
from .canon import canonical_json, derive_seed
from .color import Palette

RECIPE_VERSION = 1
PRESETS: dict[str, list[str]] = {
    "ember": ["#0d0a12", "#2b1b3d", "#7a2f5c", "#e0603f", "#ffd08a"],
    "tide": ["#04121f", "#0b3a53", "#1d7a8c", "#63c7b3", "#e9f6ef"],
    "moss": ["#0b1410", "#1c3324", "#3f6b45", "#8fbf6f", "#eaf0d8"],
    "ink": ["#f2efe9", "#b8c0cc", "#5b6b8c", "#28324d", "#0d1220"],
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


def random_recipe(
    generator_name: str | None = None,
    rng: Any = None,
    **overrides: Any,
) -> Recipe:
    """Build a recipe with randomized parameters inside their valid ranges."""
    from .rng import random_seed, stream, uniform

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
            low = spec.minimum if spec.minimum is not None else float(spec.default) - 1.0
            high = spec.maximum if spec.maximum is not None else float(spec.default) + 1.0
            # Bias toward the declared default so random recipes stay usable.
            value = uniform(seed_source, low, high)
            middle = float(spec.default)
            value = 0.65 * value + 0.35 * middle
            params[spec.name] = (
                round(value) if spec.kind == "int" else float(np.clip(value, low, high))
            )
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
