"""The systemd user units.

Kept as files rather than generated strings so they can be read, diffed, and
edited like any other part of the project. `install` copies them into the
systemd user directory and enables the timer.

Two properties matter in the unit itself. It runs as a *user* unit, so it can
reach the session's wallpaper settings; and it is `Persistent`, so a laptop that
was asleep at 8am still gets a new wallpaper on the next boot rather than
skipping that day.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UNIT_DIR = PROJECT_ROOT / "systemd"
TIMER_STEM = "genesis-daily"
SERVICE_NAME = f"{TIMER_STEM}.service"
TIMER_NAME = f"{TIMER_STEM}.timer"
UNIT_NAMES = [SERVICE_NAME, TIMER_NAME]

SYSTEMD_USER_DIR = Path.home() / ".config" / "systemd" / "user"


def unit_path(name: str) -> Path:
    return UNIT_DIR / name


def install_commands() -> list[str]:
    """The commands `install` runs, for printing when systemd is unavailable."""
    return [
        f"mkdir -p {SYSTEMD_USER_DIR}",
        f"cp {UNIT_DIR / SERVICE_NAME} {SYSTEMD_USER_DIR}",
        f"cp {UNIT_DIR / TIMER_NAME} {SYSTEMD_USER_DIR}",
        "systemctl --user daemon-reload",
        f"systemctl --user enable --now {TIMER_NAME}",
    ]


def install(destination: Path | None = None) -> list[str]:
    """Copy the units into the systemd user directory. Returns the names copied."""
    if shutil.which("systemctl") is None:
        raise OSError("systemctl is not available")
    target_dir = destination or SYSTEMD_USER_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in UNIT_NAMES:
        shutil.copyfile(unit_path(name), target_dir / name)
    return list(UNIT_NAMES)


def is_installed(destination: Path | None = None) -> bool:
    target_dir = destination or SYSTEMD_USER_DIR
    return all((target_dir / name).exists() for name in UNIT_NAMES)


def timer_status() -> tuple[bool, str]:
    """Whether the timer is enabled, and whatever systemd reports."""
    completed = subprocess.run(
        ["systemctl", "--user", "is-enabled", TIMER_NAME],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() == "enabled", completed.stdout.strip()


def next_run() -> str:
    """The timer's next scheduled run, or a note that systemd did not say."""
    completed = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            TIMER_NAME,
            "--property=NextElapseUSecRealtime",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    value = completed.stdout.strip().split("=", 1)[-1]
    return value or "unknown"