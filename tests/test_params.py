"""Parameter schema validation."""

from __future__ import annotations

import pytest

from genesis.params import ParamSpec, Requirement, Schema, build_schema


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


class TestRequirement:
    def test_below_accepts_values_under_the_limit(self) -> None:
        requirement = Requirement(param="zoom", below=1.0)
        assert requirement.satisfied_by({"zoom": 0.5})
        assert not requirement.satisfied_by({"zoom": 1.0})

    def test_above_accepts_values_over_the_limit(self) -> None:
        requirement = Requirement(param="exposure", above=1.0)
        assert requirement.satisfied_by({"exposure": 1.2})
        assert not requirement.satisfied_by({"exposure": 1.0})

    def test_equals_requires_an_exact_match(self) -> None:
        requirement = Requirement(param="fit", equals="fixed")
        assert requirement.satisfied_by({"fit": "fixed"})
        assert not requirement.satisfied_by({"fit": "auto"})

    def test_missing_or_non_numeric_values_do_not_satisfy_numeric_limits(self) -> None:
        requirement = Requirement(param="zoom", below=1.0)
        assert not requirement.satisfied_by({})
        assert not requirement.satisfied_by({"zoom": "near"})

    def test_unavailable_reasons_name_the_unmet_condition(self) -> None:
        spec = ParamSpec(
            "pan_x",
            "float",
            0.0,
            requires=(Requirement(param="zoom", below=1.0),),
        )
        assert spec.is_available({"zoom": 0.5})
        assert spec.unavailable_reason({"zoom": 0.5}) is None
        assert not spec.is_available({"zoom": 1.0})
        assert spec.unavailable_reason({"zoom": 1.0}) == "needs zoom below 1"

    def test_multiple_requirements_are_combined(self) -> None:
        spec = ParamSpec(
            "detail",
            "int",
            1,
            requires=(
                Requirement(param="mode", equals="vector"),
                Requirement(param="size", above=100),
            ),
        )
        assert spec.is_available({"mode": "vector", "size": 192})
        assert not spec.is_available({"mode": "vector", "size": 90})
        assert spec.unavailable_reason({"mode": "raster", "size": 90}) == (
            "needs mode set to vector and needs size above 100"
        )

    def test_requirements_are_serialized_for_interfaces(self) -> None:
        payload = ParamSpec(
            "extent",
            "float",
            2.0,
            requires=(Requirement(param="fit", equals="fixed"),),
        ).to_json()["requires"]
        assert payload == [
            {"param": "fit", "equals": "fixed", "below": None, "above": None}
        ]


class TestConditionalControlsAreDeclared:
    """Conditional controls must say when they are inert.

    These tests pin the currently known dependencies. They do not infer rules
    from renderer code; a newly conditional control needs its own explicit test.
    """

    def _attractor(self):
        from genesis import registry

        registry.ensure_loaded()
        return registry.get("attractor").schema

    def test_attractor_pans_require_a_cropped_view(self) -> None:
        schema = self._attractor()
        defaults = schema.defaults()
        for name in ("pan_x", "pan_y"):
            spec = schema.get(name)
            assert not spec.is_available(defaults)
            assert spec.unavailable_reason(defaults) == "needs zoom below 1"
            assert spec.is_available({**defaults, "zoom": 0.5})

    def test_attractor_extent_requires_fixed_framing(self) -> None:
        schema = self._attractor()
        defaults = schema.defaults()
        spec = schema.get("extent")
        assert not spec.is_available(defaults)
        assert spec.unavailable_reason(defaults) == "needs fit set to fixed"
        assert spec.is_available({**defaults, "fit": "fixed"})

    def test_requirements_name_parameters_in_the_same_schema(self) -> None:
        from genesis import registry

        registry.ensure_loaded()
        for generator in registry.all_generators():
            names = {spec.name for spec in generator.schema}
            for spec in generator.schema:
                for requirement in spec.requires:
                    assert requirement.param in names, (
                        f"{generator.name}.{spec.name} requires unknown "
                        f"{requirement.param!r}"
                    )
