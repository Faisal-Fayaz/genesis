"""The pinned environment must actually be the one in use.

Determinism is the product here, and byte-identical output depends on library
versions: Pillow's resampling and some encodings are not byte-stable across
releases, and numpy's PCG64 stream is guaranteed but other generators are not.
The pins in requirements.txt are therefore load-bearing, not documentation.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import numpy
import PIL
import pytest

ROOT = Path(__file__).resolve().parent.parent


def pinned_requirements() -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "==" in line:
            name, _, version = line.partition("==")
            pins[name.strip().lower()] = version.strip()
    return pins


def declared_dependencies() -> dict[str, str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    return {
        name.split("=")[0].strip().lower(): name.split("==")[1].strip()
        for name in data["project"]["dependencies"]
        if "==" in name
    }


class TestPinsAreDeclared:
    def test_requirements_file_parses(self) -> None:
        assert pinned_requirements(), "no pinned requirements found"

    def test_every_runtime_dependency_is_pinned_exactly(self) -> None:
        for name, version in declared_dependencies().items():
            assert name in pinned_requirements(), f"{name} missing from requirements.txt"
            assert pinned_requirements()[name] == version, (
                f"{name} pinned to {pinned_requirements()[name]} in requirements.txt "
                f"but {version} in pyproject.toml"
            )

    @pytest.mark.parametrize("name", ["numpy", "pillow"])
    def test_installed_version_matches_the_pin(self, name: str) -> None:
        pins = pinned_requirements()
        if name not in pins:
            pytest.skip(f"{name} is not pinned")
        installed = {"numpy": numpy.__version__, "pillow": PIL.__version__}[name]
        assert installed == pins[name], (
            f"{name} {installed} is installed but {pins[name]} is pinned. "
            "Render digests are tied to these versions."
        )