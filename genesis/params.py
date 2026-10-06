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
        return payload


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
