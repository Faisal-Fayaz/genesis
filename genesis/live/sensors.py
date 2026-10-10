"""Outside-world sensors for the live loop: screen lock and battery state.

The frame loop knows about the pointer and its own budget, but nothing about
the world around it: rendering behind a lock screen or at full rate on
battery is pure waste. These monitors close that gap with no new Python
dependencies — `gdbus` ships with the same desktop the wallpaper runs on.

Two rules govern every probe here. Fail open: a missing tool, a failed call,
or unparseable output yields `None` (unknown), and the service treats unknown
as "carry on" — a sensor must never pause a healthy session. And injectable:
each monitor takes a `run` callable so tests script every outcome without a
bus, a battery, or a lock screen.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence

# `run` takes an argv list and returns (returncode, stdout).
Runner = Callable[[Sequence[str]], tuple[int, str]]


def _run_command(argv: Sequence[str]) -> tuple[int, str]:
    """The real runner: `gdbus`, captured, never raising."""
    try:
        completed = subprocess.run(
            list(argv), capture_output=True, text=True, check=False, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return 127, ""
    return completed.returncode, completed.stdout


def _parse_bool(output: str) -> bool | None:
    """Read a D-Bus boolean out of `gdbus` output.

    ScreenSaver answers `(true,)`; UPower wraps its variant as `(<true>,)`.
    Anything else is unknown rather than False: guessing "unlocked" from
    garbage would be a behaviour, and behaviours need evidence.
    """
    text = output.strip().lower()
    if "false" in text:
        return False
    if "true" in text:
        return True
    return None


class LockMonitor:
    """Whether the session is screen-locked, via GNOME ScreenSaver."""

    def __init__(self, run: Runner | None = None) -> None:
        self._run = run or _run_command

    def locked(self) -> bool | None:
        try:
            code, out = self._run(
                [
                    "gdbus", "call", "--session",
                    "--dest", "org.gnome.ScreenSaver",
                    "--object-path", "/org/gnome/ScreenSaver",
                    "--method", "org.gnome.ScreenSaver.GetActive",
                ]
            )
        except Exception:  # noqa: BLE001 - a sensor must never kill the loop
            return None
        if code != 0:
            return None
        return _parse_bool(out)


class PowerMonitor:
    """Whether the machine is on battery, via UPower."""

    def __init__(self, run: Runner | None = None) -> None:
        self._run = run or _run_command

    def on_battery(self) -> bool | None:
        try:
            code, out = self._run(
                [
                    "gdbus", "call", "--system",
                    "--dest", "org.freedesktop.UPower",
                    "--object-path", "/org/freedesktop/UPower",
                    "--method", "org.freedesktop.DBus.Properties.Get",
                    "org.freedesktop.UPower", "OnBattery",
                ]
            )
        except Exception:  # noqa: BLE001 - a sensor must never kill the loop
            return None
        if code != 0:
            return None
        return _parse_bool(out)
