"""Reproducibility is the product. These tests are the contract."""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pytest

from genesis import registry, render
from genesis.recipe import Recipe, random_recipe
from genesis.rng import stream

# Schemas reject unknown keys, so each generator gets only its own parameters,
# all chosen at the low end of their ranges to keep tests fast. Values must stay
# within each schema's declared bounds.
TINY_PARAMS: dict[str, dict[str, int]] = {
    "attractor": {"points": 20000, "warmup": 50, "supersample": 1},
    "flowfield": {"trails": 1000, "steps": 30, "supersample": 1},
}

def tiny(generator: str, **overrides: object) -> Recipe:
    params = dict(TINY_PARAMS[generator])
    params.update(overrides)
    return Recipe(
        generator=generator,
        params=params,
        width=96,
        height=54,
        seed=1234,
    )


@pytest.fixture(scope="module", autouse=True)
def loaded() -> None:
    registry.ensure_loaded()


@pytest.mark.parametrize("name", sorted(registry.names()) if registry.names() else ["attractor"])
def test_same_recipe_is_byte_identical(name: str) -> None:
    recipe = tiny(name)
    first = render.render(recipe).digest
    second = render.render(recipe).digest
    assert first == second


@pytest.mark.parametrize("name", ["attractor", "flowfield"])
def test_seed_changes_output(name: str) -> None:
    base = tiny(name)
    other = base.with_changes(seed=99)
    assert render.render(base).digest != render.render(other).digest


@pytest.mark.parametrize("name", ["attractor", "flowfield"])
def test_empty_seed_still_reproduces(name: str) -> None:
    """An unpinned seed is derived from content, so it still reproduces."""
    first = render.render(tiny(name).with_changes(seed=None))
    second = render.render(tiny(name).with_changes(seed=None))
    assert first.digest == second.digest


@pytest.mark.parametrize("name", ["attractor", "flowfield"])
def test_unpinned_seed_tracks_content(name: str) -> None:
    """Changing content must change the derived seed."""
    recipe = tiny(name).with_changes(seed=None)
    assert recipe.seed is None
    assert recipe.effective_seed != recipe.with_changes(palette_name="solar").effective_seed
    assert recipe.effective_seed != recipe.with_changes(width=97).effective_seed


@pytest.mark.parametrize("name", ["attractor", "flowfield"])
def test_output_shape_and_finiteness(name: str) -> None:
    result = render.render(tiny(name))
    assert result.image.size == (96, 54)
    assert result.image.mode == "RGB"
    data = np.asarray(result.image)
    assert data.dtype == np.uint8
    assert data.min() >= 0 and data.max() <= 255
    # A blank canvas would render identically for every seed; guard against it.
    assert data.std() > 1.0


@pytest.mark.parametrize("name", ["attractor", "flowfield"])
def test_field_order_in_recipe_does_not_matter(name: str) -> None:
    """Canonicalization is what makes recipes diffable and hashes stable."""
    recipe = tiny(name)
    reordered = Recipe.from_json(recipe.to_json())
    assert recipe.canonical() == reordered.canonical()
    assert render.render(recipe).digest == render.render(reordered).digest


@contextmanager
def _with_stub(generator_name: str, stub):
    """Swap in a broken render function, then put the real one back."""
    generator = registry.get(generator_name)
    original = generator.render
    generator.render = stub
    try:
        yield
    finally:
        generator.render = original


def test_generator_rejects_wrong_size() -> None:
    """A generator returning the wrong shape must fail loudly."""
    with _with_stub(
        "attractor", lambda params, seed, width, height, palette: np.zeros((10, 10, 3))
    ), pytest.raises(ValueError, match="returned"):
        render.render(tiny("attractor"))


def test_generator_rejects_non_finite() -> None:
    with _with_stub(
        "attractor",
        lambda params, seed, width, height, palette: np.full((54, 96, 3), np.nan),
    ), pytest.raises(ValueError, match="non-finite"):
        render.render(tiny("attractor"))


def test_pinned_seed_survives_other_field_edits() -> None:
    """An explicit seed must not move when unrelated fields change."""
    recipe = tiny("attractor")
    edited = recipe.with_changes(palette_name="solar")
    assert recipe.effective_seed == edited.effective_seed


def test_random_recipes_are_reproducible() -> None:
    first = random_recipe("attractor", rng=np.random.default_rng(5), width=64, height=36)
    second = random_recipe("attractor", rng=np.random.default_rng(5), width=64, height=36)
    assert first.to_dict() == second.to_dict()
    assert render.render(first).digest == render.render(second).digest


def test_substreams_are_independent() -> None:
    """Drawing from one named stream must not shift another."""
    a_before = stream(99, "alpha").random(5)
    stream(99, "beta").random(1000)
    a_after = stream(99, "alpha").random(5)
    assert np.array_equal(a_before, a_after)
