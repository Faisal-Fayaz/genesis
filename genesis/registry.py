"""Generator registry.

Generators register a name, a parameter schema, and a render function. That
registry is the single source of truth for what `genesis list` prints, what the
CLI accepts, what the web form builds, and what recipe validation checks
against. Adding a generator to the package is enough to make it appear
everywhere.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from .params import Schema

RenderFn = Callable[..., Any]


class Generator:
    def __init__(self, name: str, summary: str, schema: Schema, render: RenderFn, family: str) -> None:
        self.name = name
        self.summary = summary
        self.schema = schema
        self.render = render
        self.family = family

    def __repr__(self) -> str:
        return f"Generator(name={self.name!r}, family={self.family!r})"

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "summary": self.summary,
            "family": self.family,
            "params": self.schema.to_json(),
        }


_REGISTRY: dict[str, Generator] = {}


def register(name: str, summary: str, schema: Schema, render: RenderFn, family: str) -> Generator:
    if name in _REGISTRY:
        raise ValueError(f"generator {name!r} is already registered")
    if not name.replace("_", "").isalnum():
        raise ValueError(f"generator name {name!r} must be alphanumeric")
    generator = Generator(name, summary, schema, render, family)
    _REGISTRY[name] = generator
    return generator


def get(name: str) -> Generator:
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "none"
        raise KeyError(f"unknown generator {name!r}; available: {known}") from None


def names() -> list[str]:
    return sorted(_REGISTRY)


def all_generators() -> list[Generator]:
    return [_REGISTRY[name] for name in names()]


def __iter__() -> Iterator[Generator]:
    return iter(all_generators())


def ensure_loaded() -> None:
    """Import the generator modules so their registrations happen."""
    if _REGISTRY:
        return
    from . import generators  # noqa: F401
