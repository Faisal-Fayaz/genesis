"""The render path: recipe in, image out.

This module owns the guarantee that matters most. It resolves the generator and
palette from a validated recipe, renders, applies post processing, converts to
8-bit, and records a hash of the result. Nothing else in the package is
permitted to introduce entropy, so the recorded hash is a complete fingerprint
of the output.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from . import post as post_module
from . import registry
from .color import Palette
from .generators._common import to_image
from .recipe import Recipe


@dataclass(frozen=True)
class RenderResult:
    recipe: Recipe
    image: Image.Image
    digest: str

    @property
    def size(self) -> tuple[int, int]:
        return self.image.size


def sha256_of(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def render(recipe: Recipe) -> RenderResult:
    """Render a recipe to an image plus the hash of its pixels."""
    registry.ensure_loaded()
    recipe.validate()

    generator = registry.get(recipe.generator)
    palette: Palette = recipe.effective_palette()
    seed = recipe.effective_seed

    linear = generator.render(recipe.params, seed, recipe.width, recipe.height, palette)
    linear = np.asarray(linear, dtype=np.float64)
    if linear.shape != (recipe.height, recipe.width, 3):
        raise ValueError(
            f"{recipe.generator} returned {linear.shape}, expected "
            f"{(recipe.height, recipe.width, 3)}"
        )
    if not np.isfinite(linear).all():
        raise ValueError(f"{recipe.generator} produced non-finite pixels")

    linear = post_module.apply(linear, seed, recipe.resolved_post())
    return RenderResult(recipe, to_image(linear), sha256_of(linear))


def render_to_file(recipe: Recipe, path: str | Path) -> RenderResult:
    """Render and write a PNG.

    The write is atomic via a temporary sibling file, so a reader (or a
    wallpaper swap) never observes a half-written image.
    """
    result = render(recipe)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}.{result.digest[:12]}.tmp")
    result.image.save(staging, format="PNG", optimize=True)
    staging.replace(target)
    return result


def preview_size(recipe: Recipe, longest_edge: int = 640) -> Recipe:
    """A proportionally scaled copy for fast gallery and CLI previews."""
    scale = longest_edge / max(recipe.width, recipe.height)
    return recipe.scaled(scale) if scale < 1.0 else recipe


def describe(recipe: Recipe) -> dict[str, Any]:
    """Metadata about a recipe without rendering it."""
    registry.ensure_loaded()
    generator = registry.get(recipe.generator)
    return {
        "generator": recipe.generator,
        "family": generator.family,
        "summary": generator.summary,
        "width": recipe.width,
        "height": recipe.height,
        "palette_name": recipe.palette_name,
        "palette": recipe.effective_palette().colors,
        "seed": recipe.effective_seed,
        "theme": recipe.theme,
        "slug": recipe.slug(),
        "params": recipe.params,
    }
