"""Curated recipes: a small, hand-picked set that is known to look good.

The random draw in `recipe.py` varies parameters inside caps chosen to avoid
rendering something bad. That is a floor, not a ceiling: it produces a
*variation of a decent image*, and it cannot reach the compositions that only
come from picking `scale` and `turns` against each other by eye.

This module holds those picks. Each one names a `mood` so the interface can
offer a choice without asking the user to understand 37 parameters, and each one
records *why* it is here, so the next edit can tell a deliberate look from an
accident.

Two rules were applied to every preset, both learned from reviewing contact
sheets of renders against measured pixel statistics:

- The background must stay dark. The renderer fills the canvas with palette stop
  zero, so a palette whose stops run light-to-dark yields a white background and
  desktop icons become unreadable. `ink` originally did exactly this.
- Softness comes from `bloom`, `saturation`, and `vignette`, not from
  indiscriminately lifting black or pushing contrast down. A post grade acts on
  the background too, so any hazy treatment is checked against the rendered
  background rather than trusted from its parameter values.

Presets are deliberately not exhaustive: each mood has a handful of distinct
compositions rather than a spread across the whole parameter space, because the
space contains a lot of near-duplicates.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .recipe import Recipe

# The moods the interface offers. Kept short because they are the only choice a
# user should have to make before they can get something they like.
MOODS = ("calm", "rich", "soft")


@dataclass(frozen=True)
class Curated:
    """One hand-picked composition.

    `params` is merged over the generator's defaults, `post` over the post
    chain's, so a preset only has to state what it changes.
    """
    name: str
    mood: str
    palette: str
    seed: int
    params: dict[str, Any]
    post: dict[str, Any] = None  # type: ignore[assignment]
    note: str = ""
    generator: str = "flowfield"

    def recipe(self, width: int = 1920, height: int = 1080) -> Recipe:
        """Build the recipe at a requested size.

        Ink scales with canvas size inside the generator, so a preset keeps its
        composition across resolutions instead of being tuned per size.
        """
        return Recipe(
            generator=self.generator,
            params=dict(self.params),
            width=width,
            height=height,
            palette_name=self.palette,
            seed=self.seed,
            post=dict(self.post or {}),
        )

    def label(self) -> str:
        return self.name.replace("-", " ")


# Shared shapes. Every preset is one of these three compositions at a different
# scale and turn count, which is what keeps the moods distinguishable without
# making fifteen unrelated images.
_FLOWING = {"swirl": True, "detail": 5}
_CONTOUR = {"swirl": True, "detail": 5, "trails": 6000, "steps": 230}
_SOFT = {"swirl": True, "detail": 4, "trails": 4200, "steps": 230}


CURATED: tuple[Curated, ...] = (
    # --- calm: slow, broad, low contrast. Reads as a quiet background. -------
    Curated(
        name="still-water",
        mood="calm",
        palette="tide",
        seed=4101,
        params={**_FLOWING, "scale": 0.9, "turns": 2.0, "step_size": 0.0060,
                "exposure": 1.15, "trails": 3200, "steps": 160},
        note="the widest field in the set; almost no visible structure at 1:1",
    ),
    Curated(
        name="deep-current",
        mood="calm",
        palette="slate",
        seed=4102,
        params={**_FLOWING, "scale": 0.7, "turns": 1.5, "step_size": 0.0075,
                "exposure": 1.05, "trails": 3400, "steps": 165},
        note="near-neutral grey, lowest chroma of the set by design",
    ),
    Curated(
        name="low-tide",
        mood="calm",
        palette="moss",
        seed=4103,
        params={**_FLOWING, "scale": 1.6, "turns": 2.5, "step_size": 0.0045,
                "exposure": 1.1, "trails": 3800, "steps": 175},
        post={"vignette": 0.25},
        note="vignette keeps the busy edges away from the centre",
    ),
    Curated(
        name="fog-lines",
        mood="calm",
        palette="ink",
        seed=4104,
        params={**_CONTOUR, "scale": 1.1, "turns": 2.0, "step_size": 0.0055,
                "exposure": 1.0, "trails": 4500, "steps": 200},
        note="contour lines but broad and faint",
    ),
    Curated(
        name="slate-drift",
        mood="calm",
        palette="slate",
        seed=4105,
        params={**_FLOWING, "scale": 2.4, "turns": 3.0, "step_size": 0.0035,
                "exposure": 1.2, "trails": 5200, "steps": 210},
        post={"saturation": 0.8},
        note="slightly desaturated so it stays behind other windows",
    ),
    # --- rich: high chroma, tight detail, clear line structure. --------------
    Curated(
        name="silk",
        mood="rich",
        palette="tide",
        seed=4201,
        params={**_FLOWING, "scale": 4.6, "turns": 3.8, "step_size": 0.0044,
                "exposure": 1.6},
        note="the reference look; auto sub-stepping is what makes it smooth",
    ),
    Curated(
        name="silk-dense",
        mood="rich",
        palette="moss",
        seed=4202,
        params={**_FLOWING, "scale": 7.0, "turns": 5.0, "step_size": 0.0030,
                "exposure": 1.8},
        note="same family as silk at twice the frequency",
    ),
    Curated(
        name="bramble",
        mood="rich",
        palette="neon",
        seed=4203,
        params={**_CONTOUR, "scale": 9.0, "turns": 5.5, "step_size": 0.0018,
                "exposure": 1.6},
        note="densest field here; needs the step guard to stay clean",
    ),
    Curated(
        name="forge",
        mood="rich",
        palette="ember",
        seed=4204,
        params={**_FLOWING, "scale": 3.0, "turns": 4.2, "step_size": 0.0032,
                "exposure": 1.5, "trails": 8000, "steps": 230},
        post={"grain": 0.05, "grain_size": 1.3},
        note="a little grain keeps the warm palette from looking plastic",
    ),
    Curated(
        name="filigree",
        mood="rich",
        palette="solar",
        seed=4205,
        params={**_CONTOUR, "scale": 5.5, "turns": 7.0, "step_size": 0.0022,
                "exposure": 1.7},
        post={"bloom": 0.18, "bloom_radius": 30},
        note="high turn count with a little bloom to soften the crossings",
    ),
    # --- soft: hazy, low frequency, gentle edges. Softness comes from bloom and
    # saturation, never from lowering contrast or lifting blacks. -------------
    Curated(
        name="veil",
        mood="soft",
        palette="tide",
        seed=4301,
        params={**_SOFT, "scale": 1.4, "turns": 3.0, "step_size": 0.0050,
                "exposure": 1.15, "trails": 4200, "steps": 230},
        post={"bloom": 0.3, "bloom_radius": 45},
        note="bloom does the softening; blacks stay down",
    ),
    Curated(
        name="haze",
        mood="soft",
        palette="slate",
        seed=4302,
        params={**_SOFT, "scale": 2.0, "turns": 2.8, "step_size": 0.0042,
                "exposure": 1.1, "trails": 4400, "steps": 240},
        post={"bloom": 0.3, "bloom_radius": 45},
        note="the haziest of the set",
    ),
    Curated(
        name="dawn",
        mood="soft",
        palette="solar",
        seed=4303,
        params={**_SOFT, "scale": 1.8, "turns": 3.2, "step_size": 0.0045,
                "exposure": 1.25, "trails": 4300, "steps": 235},
        post={"saturation": 0.85},
        note="warm and pale without going bright enough to fight with icons",
    ),
    Curated(
        name="murmur",
        mood="soft",
        palette="moss",
        seed=4304,
        params={**_SOFT, "scale": 3.2, "turns": 4.0, "step_size": 0.0032,
                "exposure": 1.2, "trails": 4600, "steps": 245},
        post={"saturation": 0.9, "vignette": 0.2},
        note="green, mid-frequency, deliberately flat",
    ),
    Curated(
        name="frost",
        mood="soft",
        palette="ink",
        seed=4305,
        params={**_SOFT, "scale": 2.6, "turns": 3.4, "step_size": 0.0038,
                "exposure": 1.15, "trails": 4500, "steps": 235},
        post={"bloom": 0.22, "bloom_radius": 50, "saturation": 0.7},
        note="cool blue-grey; the most desaturated preset",
    ),
    # --- splash: billowing ink clouds on black, after the ink-in-water
    # reference. Same discipline as above: dark background, quiet corners,
    # saturated cores. Triangular color picks already favor mid stops, so
    # these tune coverage and exposure only. --------------------------------
    Curated(
        name="nebula",
        mood="rich",
        palette="neon",
        seed=4401,
        generator="splash",
        params={"blobs": 45, "spread": 0.11, "turbulence": 0.45,
                "detail": 4, "color_count": 4, "fray": 0.9, "exposure": 1.2},
        note="magenta/cyan ink clouds; the reference look for splash",
    ),
    Curated(
        name="ember-cloud",
        mood="rich",
        palette="ember",
        seed=4402,
        generator="splash",
        params={"blobs": 55, "spread": 0.10, "turbulence": 0.5,
                "detail": 4, "color_count": 4, "fray": 0.9, "exposure": 1.1},
        note="warm red/gold smoke, denser than nebula",
    ),
    Curated(
        name="night-fog",
        mood="calm",
        palette="slate",
        seed=4403,
        generator="splash",
        params={"blobs": 40, "spread": 0.12, "turbulence": 0.4,
                "detail": 4, "color_count": 3, "fray": 0.8, "exposure": 1.0},
        note="grey smoke wisps; the quietest splash",
    ),
    Curated(
        name="dusk-ash",
        mood="soft",
        palette="solar",
        seed=4404,
        generator="splash",
        params={"blobs": 38, "spread": 0.11, "turbulence": 0.45,
                "detail": 4, "color_count": 3, "fray": 0.9, "exposure": 0.95},
        post={"saturation": 0.85},
        note="warm haze held dim enough for desktop icons",
    ),
)


def _index() -> dict[str, Curated]:
    return {preset.name: preset for preset in CURATED}


def get(name: str) -> Curated:
    """Look up a curated preset by name."""
    try:
        return _index()[name]
    except KeyError:
        known = ", ".join(sorted(_index()))
        raise KeyError(f"unknown preset {name!r}; known presets are {known}") from None


def names() -> list[str]:
    return [preset.name for preset in CURATED]


def for_mood(mood: str) -> list[Curated]:
    """Every curated preset tagged with a mood, in declaration order."""
    return [preset for preset in CURATED if preset.mood == mood]


def moods_of(presets: Iterable[Curated]) -> list[str]:
    """The moods represented in an arbitrary selection, in canonical order."""
    present = {preset.mood for preset in presets}
    return [mood for mood in MOODS if mood in present]