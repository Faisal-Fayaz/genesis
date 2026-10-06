"""Recipe parsing, validation, and round-tripping."""

from __future__ import annotations

import json

import pytest

from genesis import registry
from genesis.recipe import PRESETS, Recipe


def base(**overrides) -> Recipe:
    fields = {
        "generator": "attractor",
        "params": {"points": 50000},
        "width": 64,
        "height": 36,
        "seed": 5,
    }
    fields.update(overrides)
    return Recipe(**fields)


def test_round_trip_through_json() -> None:
    recipe = base()
    assert Recipe.from_json(recipe.to_json()).to_dict() == recipe.to_dict()


def test_round_trip_through_file(tmp_path) -> None:
    recipe = base(post={"grain": 0.03}, theme="light")
    path = recipe.save(tmp_path / "nested" / "r.json")
    assert Recipe.load(path).to_dict() == recipe.to_dict()


def test_saved_file_is_readable_json() -> None:
    text = base().to_json()
    assert json.loads(text)["generator"] == "attractor"
    assert text.endswith("\n")


def test_missing_params_take_defaults() -> None:
    recipe = Recipe(generator="flowfield", width=64, height=36)
    assert recipe.params["trails"] == registry.get("flowfield").schema.get("trails").default


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"width": 4}, "width"),
        ({"height": 99999}, "height"),
        ({"theme": "neon"}, "theme"),
        ({"seed": "abc"}, "seed"),
        ({"seed": True}, "seed"),
        ({"version": 99}, "version"),
        ({"params": {"points": -5}}, "points"),
        ({"params": {"form": "nope"}}, "form"),
        ({"params": {"nope": 1}}, "unknown parameter"),
        ({"post": {"nope": 1}}, "unknown post option"),
        ({"palette": []}, "palette"),
    ],
)
def test_invalid_recipes_are_rejected(overrides, expected) -> None:
    with pytest.raises((ValueError, KeyError), match=expected):
        base(**overrides)


def test_unknown_generator_lists_alternatives() -> None:
    with pytest.raises(KeyError, match="available"):
        Recipe(generator="nonexistent")


def test_unknown_recipe_field_is_rejected() -> None:
    payload = base().to_dict()
    payload["surprise"] = 1
    with pytest.raises(ValueError, match="unknown recipe field"):
        Recipe.from_dict(payload)


def test_malformed_json_is_reported_clearly() -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        Recipe.from_json("{not json")


def test_canonical_form_ignores_key_order() -> None:
    recipe = base()
    reordered = dict(reversed(list(recipe.to_dict().items())))
    assert recipe.canonical() == Recipe.from_dict(reordered).canonical()


def test_with_changes_does_not_mutate_original() -> None:
    recipe = base()
    before = recipe.to_dict()
    recipe.with_changes(width=128, params={"points": 20000})
    assert recipe.to_dict() == before


def test_scaled_keeps_aspect() -> None:
    assert base(width=1920, height=1080).scaled(0.5).width == 960
    assert base(width=1920, height=1080).scaled(0.5).height == 540


def test_palette_falls_back_to_preset() -> None:
    recipe = base(palette_name="ember")
    assert recipe.effective_palette().colors == PRESETS["ember"]


def test_explicit_palette_wins() -> None:
    recipe = base(palette_name="ember", palette=["#000000", "#ffffff"])
    assert recipe.effective_palette().colors == ["#000000", "#ffffff"]


def test_unknown_palette_name_falls_back() -> None:
    assert base(palette_name="nonexistent").effective_palette().colors == PRESETS["tide"]


def test_slug_is_stable_and_descriptive() -> None:
    slug = base(width=320, height=180, seed=0x1234).slug()
    assert slug == "attractor-320x180-00001234"


def test_all_presets_are_valid_hex() -> None:
    for name, colors in PRESETS.items():
        assert len(colors) >= 3, name
        for color in colors:
            assert len(color) == 7 and color.startswith("#")
            int(color[1:], 16)
