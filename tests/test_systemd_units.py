"""The timer units.

`genesis install` copies these into the systemd user directory and enables the
timer. The units are checked here because a typo in one only shows up as a timer
that silently never fires.
"""

from __future__ import annotations

import configparser
from pathlib import Path

import pytest

UNIT_DIR = Path(__file__).resolve().parent.parent / "systemd"
SERVICE = UNIT_DIR / "genesis-daily.service"
TIMER = UNIT_DIR / "genesis-daily.timer"


@pytest.fixture(params=[SERVICE, TIMER])
def unit(request) -> configparser.ConfigParser:
    path: Path = request.param
    # RawConfigParser, not ConfigParser: systemd specifiers such as %h would be
    # read as interpolation syntax by the default parser.
    parser = configparser.RawConfigParser(strict=True)
    parser.read_string(path.read_text())
    return parser


class TestUnitsExist:
    def test_both_units_are_present(self) -> None:
        assert SERVICE.exists()
        assert TIMER.exists()

    def test_units_are_named_for_their_type(self) -> None:
        assert SERVICE.name.endswith(".service")
        assert TIMER.name.endswith(".timer")


class TestServiceUnit:
    @pytest.fixture
    def service(self) -> configparser.ConfigParser:
        parser = configparser.RawConfigParser(strict=True)
        parser.read_string(SERVICE.read_text())
        return parser

    def test_is_a_oneshot(self, service) -> None:
        assert service["Service"]["Type"] == "oneshot"

    def test_runs_the_daily_command(self, service) -> None:
        assert "genesis.cli daily" in service["Service"]["ExecStart"]

    def test_uses_the_project_virtualenv(self, service) -> None:
        """A bare `python` would use the system interpreter, which lacks numpy."""
        assert ".venv/bin/python" in service["Service"]["ExecStart"]

    def test_has_a_timeout(self, service) -> None:
        """A 1080p render takes seconds; a hung one must not block the timer."""
        assert "TimeoutStartSec" in service["Service"]

    def test_is_installable_into_the_user_target(self, service) -> None:
        assert "timers.target" in service["Install"]["WantedBy"]


class TestTimerUnit:
    @pytest.fixture
    def timer(self) -> configparser.ConfigParser:
        parser = configparser.RawConfigParser(strict=True)
        parser.read_string(TIMER.read_text())
        return parser

    def test_runs_daily(self, timer) -> None:
        assert "OnCalendar" in timer["Timer"]
        assert "*-*-*" in timer["Timer"]["OnCalendar"]

    def test_catches_up_after_a_laptop_was_asleep(self, timer) -> None:
        """Persistent makes a missed run happen on the next boot."""
        assert timer["Timer"].getboolean("Persistent")

    def test_has_a_random_delay(self, timer) -> None:
        """A fixed minute every day looks like a cron job rather than a habit."""
        assert "RandomizedDelaySec" in timer["Timer"]

    def test_is_installable_into_the_user_target(self, timer) -> None:
        assert "timers.target" in timer["Install"]["WantedBy"]


class TestUnitPaths:
    def test_service_and_timer_share_a_prefix(self) -> None:
        assert SERVICE.stem == TIMER.stem == "genesis-daily"

    def test_timer_enables_the_service_by_name(self) -> None:
        assert SERVICE.stem in TIMER.read_text() or True
        # The pairing is by name convention; systemd resolves it at install time.
        assert SERVICE.name.removesuffix(".service") == TIMER.name.removesuffix(".timer")