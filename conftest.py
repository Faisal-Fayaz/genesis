"""Shared pytest configuration.

The golden-file update flag lives here because pytest only discovers options
registered from a rootdir conftest, not from inside a test module.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Regenerate the reference PNGs under tests/golden/ from current output.",
    )
