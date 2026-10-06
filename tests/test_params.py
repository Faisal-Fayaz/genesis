"""Parameter schema validation."""

from __future__ import annotations

import pytest

from genesis.params import ParamSpec, Schema, build_schema


def sample() -> Schema:
    return build_schema(
        [
            ParamSpec("count", "int", 10, minimum=1, maximum=100),
            ParamSpec("gain", "float", 1.0, minimum=0.0, maximum=2.0, step=0.1),
            ParamSpec("flag", "bool", False),
            ParamSpec("mode", "enum", "soft", choices=("soft", "hard")),
            ParamSpec("tint", "color", "#112233"),
        ]
    )


def test_defaults_are_complete() -> None:
    assert sample().defaults() == {
        "count": 10,
        "gain": 1.0,
        "flag": False,
        "mode": "soft",
        "tint": "#112233",
    }


def test_partial_values_merge_with_defaults() -> None:
    assert sample().validate({"count": 5})["gain"] == 1.0


def test_missing_keys_default_rather_than_fail() -> None:
    assert sample().validate({}) == sample().defaults()


@pytest.mark.parametrize(
    "values, expected",
    [
        ({"count": 0}, "below the minimum"),
        ({"count": 101}, "above the maximum"),
        ({"count": "many"}, "expected a int"),
        ({"count": True}, "booleans are not numbers"),
        ({"gain": -1}, "below the minimum"),
        ({"mode": "loud"}, "expected one of"),
        ({"mode": 3}, "expected one of"),
        ({"flag": "yes"}, "expected true or false"),
        ({"tint": "blue"}, "expected a hex color"),
        ({"tint": "#12"}, "expected a hex color"),
        ({"unknown": 1}, "unknown parameter"),
    ],
)
def test_invalid_values_are_rejected_with_context(values, expected) -> None:
    with pytest.raises(ValueError, match=expected):
        sample().validate(values)


def test_color_shorthand_expands() -> None:
    assert sample().validate({"tint": "#abc"})["tint"] == "#aabbcc"


def test_color_is_normalized_to_lowercase() -> None:
    assert sample().validate({"tint": "#AABBCC"})["tint"] == "#aabbcc"


def test_int_values_are_coerced_to_int() -> None:
    value = sample().validate({"count": 7.9})["count"]
    assert value == 7 and isinstance(value, int)


def test_membership_and_lookup() -> None:
    schema = sample()
    assert "gain" in schema
    assert "missing" not in schema
    assert schema.get("gain").maximum == 2.0
    with pytest.raises(KeyError):
        schema.get("missing")


def test_duplicate_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        build_schema([ParamSpec("a", "int", 1), ParamSpec("a", "int", 2)])


def test_enum_requires_choices() -> None:
    with pytest.raises(ValueError, match="needs choices"):
        ParamSpec("a", "enum", "x")


def test_inverted_bounds_are_rejected() -> None:
    with pytest.raises(ValueError, match="minimum exceeds maximum"):
        ParamSpec("a", "int", 1, minimum=10, maximum=1)


def test_json_form_carries_bounds_for_the_web_form() -> None:
    payload = {item["name"]: item for item in sample().to_json()}
    assert payload["count"]["minimum"] == 1
    assert payload["count"]["maximum"] == 100
    assert payload["mode"]["choices"] == ["soft", "hard"]
    assert "choices" not in payload["count"]
