"""Which attractor forms are actually usable.

Hopalong and Svensson pass the structural checks but render poorly: Hopalong's
orbits concentrate onto a handful of points, and Svensson's usable coefficient
region is tiny and fragile. Rather than let that show up as a surprise in the
gallery, the weakness is asserted here so a regression in the good forms, or a
change that would make a weak form worse, is visible.

When a form is fixed or removed, delete its entry and this file gets smaller.
"""

from __future__ import annotations

import numpy as np
import pytest

from genesis import render
from genesis.color import linear_to_srgb
from genesis.recipe import Recipe

# Measured coverage and contrast for each form. A form not listed here is not
# covered by this check; see MODULE docstring.
FORMS = ["clifford", "dejong"]


def recipe_for(form: str, seed: int = 7) -> Recipe:
    return Recipe(
        generator="attractor",
        params={"form": form, "points": 150000, "supersample": 1, "zoom": 0.5},
        width=192,
        height=108,
        seed=seed,
        palette_name="tide",
    )


def render_form(form: str, seed: int = 7) -> np.ndarray:
    return np.asarray(render.render(recipe_for(form, seed)).image, dtype=np.float64)


def coverage(form: str) -> float:
    """Fraction of pixels clearly distinct from the palette floor."""
    floor = linear_to_srgb(recipe_for(form).effective_palette().to_linear(0.0)) * 255.0
    values = render_form(form)
    return float((np.abs(values - floor).sum(axis=2) > 24).mean())


def contrast(form: str) -> float:
    from PIL import Image

    image = Image.fromarray(render_form(form).astype(np.uint8)).convert("L")
    return float(np.asarray(image, dtype=np.float64).std())


@pytest.mark.parametrize("form", FORMS)
class TestUsableForms:
    def test_covers_a_meaningful_part_of_the_frame(self, form: str) -> None:
        assert coverage(form) > 0.05, f"{form} covers only {coverage(form):.3f}"

    def test_has_real_contrast(self, form: str) -> None:
        assert contrast(form) > 10.0

    def test_is_not_a_single_saturated_block(self, form: str) -> None:
        values = render_form(form)
        assert float((values.min(axis=2) >= 254).mean()) < 0.02

    def test_varies_with_the_seed(self, form: str) -> None:
        assert not np.array_equal(render_form(form, 7), render_form(form, 8))


class TestKnownWeakForms:
    """These render, but not well. Assert the floor so they cannot get worse.

    Hopalong concentrates: its orbits spend most iterations near an attracting
    point, so the image is mostly background with a few bright clusters.
    Svensson barely has a usable coefficient region at all.
    """

    @pytest.mark.parametrize("form", ["hopalong", "svensson"])
    def test_still_renders_something(self, form: str) -> None:
        values = render_form(form)
        assert values.std() > 0.5, f"{form} produced an entirely uniform frame"

    @pytest.mark.parametrize("form", ["hopalong", "svensson"])
    def test_does_not_clip_to_white_blocks(self, form: str) -> None:
        values = render_form(form)
        assert float((values.min(axis=2) >= 254).mean()) < 0.02

    def test_hopalong_is_the_weakest_of_the_four(self) -> None:
        """Records why hopalong is the first candidate for removal."""
        assert coverage("hopalong") < coverage("clifford")