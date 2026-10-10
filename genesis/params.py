"""Parameter schema.

One declarative description per knob, shared by the CLI, the web gallery's
auto-generated form, and validation. A generator that declares its parameters
here gets command-line flags and a UI for free.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

Kind = Literal["int", "float", "bool", "enum", "color", "string"]


@dataclass(frozen=True)
class Requirement:
    """A condition under which a parameter has any effect at all.

    Some parameters are inert until another one is set a particular way: the
    attractor's panning does nothing until `zoom` drops below 1.0, and `extent`
    does nothing until `fit` is "fixed". A UI that lists these without saying so
    presents controls that quietly do nothing, which is worse than not offering
    them.

    Declared rather than inferred because the generator knows the rule and the
    interface does not, and because a rule nobody wrote down cannot be checked.
    """

    param: str
    equals: Any = None
    below: float | None = None
    above: float | None = None

    def satisfied_by(self, values: dict[str, Any]) -> bool:
        """Whether `values` meets this requirement."""
        actual = values.get(self.param)
        if actual is None:
            return False
        if self.equals is not None and actual != self.equals:
            return False
        if self.below is not None:
            try:
                return float(actual) < self.below
            except (TypeError, ValueError):
                return False
        if self.above is not None:
            try:
                return float(actual) > self.above
            except (TypeError, ValueError):
                return False
        return True

    def describe(self) -> str:
        """A sentence for the interface to show next to a disabled control."""
        if self.below is not None:
            return f"needs {self.param} below {self.below:g}"
        if self.above is not None:
            return f"needs {self.param} above {self.above:g}"
        return f"needs {self.param} set to {self.equals}"


@dataclass(frozen=True)
class ParamSpec:
    name: str
    kind: Kind
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: tuple[str, ...] = ()
    label: str = ""
    help: str = ""
    requires: tuple[Requirement, ...] = ()

    def __post_init__(self) -> None:
        if self.kind == "enum" and not self.choices:
            raise ValueError(f"enum parameter {self.name!r} needs choices")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError(f"{self.name!r}: minimum exceeds maximum")

    @property
    def display_label(self) -> str:
        return self.label or self.name.replace("_", " ").title()

    def coerce(self, value: Any) -> Any:
        """Validate and normalize one raw value.

        Raises ValueError with a message aimed at whoever supplied the value,
        since that is usually a person editing a recipe file.
        """
        try:
            if self.kind in ("int", "float"):
                if isinstance(value, bool):
                    raise TypeError("booleans are not numbers here")
                try:
                    number = int(value) if self.kind == "int" else float(value)
                except (TypeError, ValueError):
                    raise TypeError(
                        f"expected a {self.kind}, got {value!r}"
                    ) from None
            elif self.kind == "bool":
                if not isinstance(value, bool):
                    raise TypeError(f"expected true or false, got {value!r}")
                number = value
            elif self.kind == "enum":
                if not isinstance(value, str):
                    raise TypeError(f"expected one of {self.choices}, got {value!r}")
                if value not in self.choices:
                    raise ValueError(f"expected one of {list(self.choices)}, got {value!r}")
                return value
            elif self.kind == "color":
                text = str(value).strip().lower()
                if not text.startswith("#"):
                    raise ValueError(f"expected a hex color like #1a2b3c, got {value!r}")
                digits = text[1:]
                if len(digits) == 3:
                    digits = "".join(char * 2 for char in digits)
                elif len(digits) != 6:
                    raise ValueError(f"expected a hex color like #1a2b3c, got {value!r}")
                int(digits, 16)
                return "#" + digits
            else:
                return str(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"parameter {self.name!r}: {exc}") from exc

        if self.kind in ("int", "float"):
            if self.minimum is not None and number < self.minimum:
                raise ValueError(
                    f"parameter {self.name!r}: {number} is below the minimum {self.minimum}"
                )
            if self.maximum is not None and number > self.maximum:
                raise ValueError(
                    f"parameter {self.name!r}: {number} is above the maximum {self.maximum}"
                )
        return number

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"name": self.name, "kind": self.kind, "default": self.default}
        for key in ("minimum", "maximum", "step"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        if self.choices:
            payload["choices"] = list(self.choices)
        if self.label:
            payload["label"] = self.label
        if self.help:
            payload["help"] = self.help
        if self.requires:
            payload["requires"] = [
                {
                    "param": requirement.param,
                    "equals": requirement.equals,
                    "below": requirement.below,
                    "above": requirement.above,
                }
                for requirement in self.requires
            ]
        return payload

    def is_available(self, values: dict[str, Any]) -> bool:
        """Whether this parameter currently has any effect."""
        return all(requirement.satisfied_by(values) for requirement in self.requires)

    def unavailable_reason(self, values: dict[str, Any]) -> str | None:
        """Why this parameter is inert, or None if it is live."""
        unmet = [r for r in self.requires if not r.satisfied_by(values)]
        if not unmet:
            return None
        return " and ".join(requirement.describe() for requirement in unmet)


@dataclass(frozen=True)
class Schema:
    """An ordered set of parameter specs."""

    specs: tuple[ParamSpec, ...] = field(default_factory=tuple)

    def __iter__(self):
        return iter(self.specs)

    def __len__(self) -> int:
        return len(self.specs)

    def __contains__(self, name: object) -> bool:
        return any(spec.name == name for spec in self.specs)

    def get(self, name: str) -> ParamSpec:
        for spec in self.specs:
            if spec.name == name:
                return spec
        raise KeyError(name)

    def defaults(self) -> dict[str, Any]:
        return {spec.name: spec.default for spec in self.specs}

    def validate(self, values: dict[str, Any]) -> dict[str, Any]:
        """Coerce a full parameter set, rejecting unknown keys.

        Missing keys fall back to the declared default so recipes stay readable
        when a generator grows new parameters.
        """
        unknown = sorted(set(values) - {spec.name for spec in self.specs})
        if unknown:
            known = ", ".join(spec.name for spec in self.specs)
            raise ValueError(f"unknown parameter(s) {unknown}; known parameters are {known}")
        merged = {
            spec.name: spec.coerce(spec.default) for spec in self.specs
        }
        merged.update({key: self.get(key).coerce(value) for key, value in values.items()})
        return merged

    def to_json(self) -> list[dict[str, Any]]:
        return [spec.to_json() for spec in self.specs]


def build_schema(specs: Sequence[ParamSpec]) -> Schema:
    names = [spec.name for spec in specs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate parameter names: {duplicates}")
    return Schema(tuple(specs))
