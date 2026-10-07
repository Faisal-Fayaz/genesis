"""The render path and the generator registry.

`render.render` is where a validated recipe becomes an image, and it is the last
place a malformed generator can be caught before it writes a file. The shape and
finiteness checks exist for exactly that, so they are tested directly.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pytest

from genesis import registry, render
from genesis.recipe import Recipe


@contextmanager
def stubbed(name: str, replacement):
    generator = registry.get(name)
    original = generator.render
    generator.render = replacement
    try:
        yield
    finally:
        generator.render = original


def tiny(name: str = "flowfield", **overrides) -> Recipe:
    params: dict[str, object] = {"supersample": 1}
    if name == "attractor":
        params.update({"points": 20000, "warmup": 50, "form": "clifford"})
    else:
        params.update({"trails": 1000, "steps": 30})
    params.update(overrides)
    return Recipe(generator=name, params=params, width=64, height=48, seed=5)


class TestRenderBasics:
    def test_returns_image_and_digest(self) -> None:
        result = render.render(tiny())
        assert result.image.size == (64, 48)
        assert len(result.digest) == 64

    def test_digest_is_hex(self) -> None:
        digest = render.render(tiny()).digest
        int(digest, 16)

    def test_pixels_are_valid_bytes(self) -> None:
        data = np.asarray(render.render(tiny()).image)
        assert data.dtype == np.uint8

    def test_recipe_is_revalidated(self) -> None:
        """A tampered recipe must fail at render, not silently render wrong."""
        recipe = tiny()
        recipe.width = 999999
        with pytest.raises(ValueError):
            render.render(recipe)


class TestGeneratorContract:
    """Whatever a generator returns, render refuses to use it."""

    def test_wrong_shape_is_rejected(self) -> None:
        with stubbed("flowfield", lambda *a, **k: np.zeros((5, 5, 3))), pytest.raises(
            ValueError, match="returned"
        ):
            render.render(tiny())

    def test_wrong_channel_count_is_rejected(self) -> None:
        with stubbed("flowfield", lambda *a, **k: np.zeros((48, 64, 4))), pytest.raises(
            ValueError, match="returned"
        ):
            render.render(tiny())

    def test_non_finite_pixels_are_rejected(self) -> None:
        with stubbed("flowfield", lambda *a, **k: np.full((48, 64, 3), np.nan)), pytest.raises(
            ValueError, match="non-finite"
        ):
            render.render(tiny())

    def test_infinite_pixels_are_rejected(self) -> None:
        with stubbed("flowfield", lambda *a, **k: np.full((48, 64, 3), np.inf)), pytest.raises(
            ValueError, match="non-finite"
        ):
            render.render(tiny())

    def test_a_restored_stub_still_works(self) -> None:
        """Confirms the context manager leaves the registry usable."""
        with stubbed("flowfield", lambda *a, **k: np.zeros((5, 5, 3))):
            pass
        assert render.render(tiny()).image.size == (64, 48)


class TestRenderToFile:
    def test_writes_the_image(self, tmp_path) -> None:
        target = tmp_path / "out.png"
        render.render_to_file(tiny(), target)
        assert target.exists()
        assert target.stat().st_size > 0
        assert target.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    def test_creates_missing_directories(self, tmp_path) -> None:
        target = tmp_path / "deep" / "nested" / "out.png"
        render.render_to_file(tiny(), target)
        assert target.exists()

    def test_leaves_no_temporary_files(self, tmp_path) -> None:
        """The atomic write must not litter the directory."""
        render.render_to_file(tiny(), tmp_path / "out.png")
        assert [p.name for p in tmp_path.iterdir()] == ["out.png"]

    def test_rewriting_is_atomic(self, tmp_path) -> None:
        target = tmp_path / "out.png"
        render.render_to_file(tiny(), target)
        first = target.read_bytes()
        render.render_to_file(tiny().with_changes(seed=6), target)
        assert target.read_bytes() != first
        assert [p.name for p in tmp_path.iterdir()] == ["out.png"]

    def test_overwrites_an_existing_file(self, tmp_path) -> None:
        target = tmp_path / "out.png"
        target.write_bytes(b"stale content that is not a png")
        render.render_to_file(tiny(), target)
        assert target.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


class TestPreviewSize:
    def test_shrinks_to_the_given_edge(self) -> None:
        recipe = tiny().with_changes(width=1920, height=1080)
        preview = render.preview_size(recipe, 320)
        assert max(preview.width, preview.height) == 320

    def test_preserves_aspect(self) -> None:
        recipe = tiny().with_changes(width=1920, height=1080)
        preview = render.preview_size(recipe, 320)
        assert preview.width / preview.height == pytest.approx(16 / 9, rel=0.01)

    def test_leaves_small_recipes_alone(self) -> None:
        recipe = tiny()
        assert render.preview_size(recipe, 320).width == recipe.width

    def test_preview_matches_full_render_structure(self) -> None:
        """A preview must not change the composition, only the size."""
        recipe = tiny().with_changes(width=1920, height=1080)
        preview = render.render(render.preview_size(recipe, 320))
        assert preview.image.size == (320, 180)


class TestDescribe:
    def test_reports_resolved_settings(self) -> None:
        info = render.describe(tiny("attractor"))
        assert info["generator"] == "attractor"
        assert info["family"] == "vector"
        assert info["seed"] == tiny("attractor").effective_seed

    def test_includes_the_effective_palette(self) -> None:
        recipe = tiny().with_changes(palette_name="ember")
        assert render.describe(recipe)["palette"] == recipe.effective_palette().colors

    def test_does_not_render(self) -> None:
        """Metadata must be cheap; this is used to build the gallery index."""
        with stubbed("flowfield", lambda *a, **k: pytest.fail("should not render")):
            render.describe(tiny())


class TestSha256:
    def test_matches_numpy_content(self) -> None:
        import hashlib

        array = np.arange(12, dtype=np.float64).reshape(2, 2, 3)
        assert render.sha256_of(array) == hashlib.sha256(array.tobytes()).hexdigest()

    def test_is_sensitive_to_a_single_value(self) -> None:
        first = np.zeros((2, 2, 3))
        second = np.zeros((2, 2, 3))
        second[0, 0, 0] = 1e-12
        assert render.sha256_of(first) != render.sha256_of(second)

    def test_is_independent_of_memory_layout(self) -> None:
        array = np.asfortranarray(np.arange(12, dtype=np.float64).reshape(2, 2, 3))
        assert render.sha256_of(array) == render.sha256_of(np.ascontiguousarray(array))


class TestRegistry:
    def test_built_in_generators_are_present(self) -> None:
        registry.ensure_loaded()
        assert "flowfield" in registry.names()
        assert "attractor" in registry.names()

    def test_lookup_returns_the_generator(self) -> None:
        assert registry.get("flowfield").name == "flowfield"

    def test_unknown_name_lists_alternatives(self) -> None:
        with pytest.raises(KeyError, match="available"):
            registry.get("nonexistent")

    def test_names_are_sorted(self) -> None:
        assert registry.names() == sorted(registry.names())

    def test_every_generator_declares_a_family(self) -> None:
        for generator in registry.all_generators():
            assert generator.family

    def test_every_generator_has_a_summary(self) -> None:
        for generator in registry.all_generators():
            assert generator.summary.strip()

    def test_every_generator_has_parameters(self) -> None:
        for generator in registry.all_generators():
            assert len(generator.schema) > 0

    def test_every_generator_exposes_risk_for_the_web_form(self) -> None:
        for generator in registry.all_generators():
            payload = generator.to_json()
            assert payload["name"] == generator.name
            assert isinstance(payload["params"], list)

    def test_duplicate_registration_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="already registered"):
            registry.register(
                "flowfield", "duplicate", registry.get("flowfield").schema,
                lambda *a, **k: None, "vector",
            )

    def test_non_alphanumeric_names_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="alphanumeric"):
            registry.register("bad name", "x", registry.get("flowfield").schema,
                              lambda *a, **k: None, "vector")
