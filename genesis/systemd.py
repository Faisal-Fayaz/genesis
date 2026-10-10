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

LIVE_STEM = "genesis-live"
LIVE_SERVICE_NAME = f"{LIVE_STEM}.service"

ASSETS_DIR = PROJECT_ROOT / "assets"
DESKTOP_NAME = "genesis.desktop"
ICON_NAME = "genesis.svg"

SYSTEMD_USER_DIR = Path.home() / ".config" / "systemd" / "user"
APPLICATIONS_DIR = Path.home() / ".local" / "share" / "applications"
ICONS_DIR = Path.home() / ".local" / "share" / "icons" / "hicolor" / "scalable" / "apps"


def unit_path(name: str) -> Path:
    return UNIT_DIR / name


def install_commands(live: bool = False, app: bool = False) -> list[str]:
    """The commands `install` runs, for printing when systemd is unavailable."""
    commands = [
        f"mkdir -p {SYSTEMD_USER_DIR}",
        f"cp {UNIT_DIR / SERVICE_NAME} {SYSTEMD_USER_DIR}",
        f"cp {UNIT_DIR / TIMER_NAME} {SYSTEMD_USER_DIR}",
        "systemctl --user daemon-reload",
        f"systemctl --user enable --now {TIMER_NAME}",
    ]
    if live:
        commands[2:2] = [f"cp {UNIT_DIR / LIVE_SERVICE_NAME} {SYSTEMD_USER_DIR}"]
    if app:
        commands.extend(
            [
                f"mkdir -p {APPLICATIONS_DIR} {ICONS_DIR}",
                f"cp {ASSETS_DIR / DESKTOP_NAME} {APPLICATIONS_DIR}",
                f"cp {ASSETS_DIR / ICON_NAME} {ICONS_DIR}",
                "update-desktop-database ~/.local/share/applications || true",
            ]
        )
    return commands


def uninstall_commands() -> list[str]:
    """The commands `uninstall` runs, for printing instead of doing."""
    return [
        f"systemctl --user disable --now {TIMER_NAME}",
        (
            f"rm -f {SYSTEMD_USER_DIR / SERVICE_NAME} "
            f"{SYSTEMD_USER_DIR / TIMER_NAME} "
            f"{SYSTEMD_USER_DIR / LIVE_SERVICE_NAME}"
        ),
        f"rm -f {APPLICATIONS_DIR / DESKTOP_NAME} {ICONS_DIR / ICON_NAME}",
        "systemctl --user daemon-reload",
        "update-desktop-database ~/.local/share/applications || true",
    ]


def resolve_executable() -> str:
    """The `genesis` command services and launchers should invoke.

    Prefers a `genesis` on PATH (a real install); falls back to the current
    interpreter running this checkout's CLI, so a development tree still
    produces working units.
    """
    import sys

    found = shutil.which("genesis")
    if found:
        return found
    return f"{sys.executable} -m genesis.cli"


def render_unit(name: str, executable: str | None = None) -> str:
    """A unit file with its ExecStart pointed at the installed executable.

    The shipped units reference the development checkout, which is correct for
    this repository but wrong the moment anything moves. Rendering rewrites only
    the ExecStart command, leaving every other line byte-identical.
    """
    import re

    executable = executable or resolve_executable()
    text = unit_path(name).read_text()
    rendered, count = re.subn(
        r"^ExecStart=.*genesis\.cli (\w+)",
        f"ExecStart={executable} \\1",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise ValueError(f"unit {name} has no ExecStart invoking genesis.cli")
    return rendered


def install(
    destination: Path | None = None, live: bool = False, app: bool = False
) -> list[str]:
    """Copy the units into the systemd user directory. Returns the names copied.

    The live service is installed disabled: it is copied but never enabled
    here, so nothing animates until the user enables it explicitly.
    """
    if shutil.which("systemctl") is None:
        raise OSError("systemctl is not available")
    target_dir = destination or SYSTEMD_USER_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    names = list(UNIT_NAMES)
    if live:
        names.append(LIVE_SERVICE_NAME)
    executable = resolve_executable()
    for name in names:
        # Only services invoke the executable; the timer only schedules.
        if name.endswith(".service"):
            (target_dir / name).write_text(render_unit(name, executable))
        else:
            shutil.copyfile(unit_path(name), target_dir / name)
    if app:
        install_app_files()
        names.append(DESKTOP_NAME)
    return names


def install_app_files(
    applications_dir: Path | None = None,
    icons_dir: Path | None = None,
    executable: str | None = None,
) -> list[str]:
    """Install the desktop entry and icon. Returns the installed file names."""
    applications_dir = applications_dir or APPLICATIONS_DIR
    icons_dir = icons_dir or ICONS_DIR
    applications_dir.mkdir(parents=True, exist_ok=True)
    icons_dir.mkdir(parents=True, exist_ok=True)
    entry = (ASSETS_DIR / DESKTOP_NAME).read_text().replace(
        "@GENESIS_EXEC@", executable or resolve_executable()
    )
    (applications_dir / DESKTOP_NAME).write_text(entry)
    shutil.copyfile(ASSETS_DIR / ICON_NAME, icons_dir / ICON_NAME)
    return [DESKTOP_NAME, ICON_NAME]


def is_installed(destination: Path | None = None) -> bool:
    target_dir = destination or SYSTEMD_USER_DIR
    return all((target_dir / name).exists() for name in UNIT_NAMES)


def uninstall(
    destination: Path | None = None,
    applications_dir: Path | None = None,
    icons_dir: Path | None = None,
) -> list[str]:
    """Disable the timer and remove everything install may have put in place.

    Never raises for missing pieces: uninstalling twice, or uninstalling
    what was never installed, reports nothing removed. The library, images,
    and settings are deliberately untouched — uninstall removes the
    automation, never your wallpapers. Returns the names removed.
    """
    target_dir = destination or SYSTEMD_USER_DIR
    apps = applications_dir or APPLICATIONS_DIR
    icons = icons_dir or ICONS_DIR
    removed: list[str] = []
    have_systemctl = shutil.which("systemctl") is not None
    if have_systemctl:
        subprocess.run(
            ["systemctl", "--user", "disable", "--now", TIMER_NAME],
            capture_output=True,
            check=False,
        )
    for name in (*UNIT_NAMES, LIVE_SERVICE_NAME):
        try:
            (target_dir / name).unlink()
        except FileNotFoundError:
            pass
        else:
            removed.append(name)
    for directory, name in ((apps, DESKTOP_NAME), (icons, ICON_NAME)):
        try:
            (directory / name).unlink()
        except FileNotFoundError:
            pass
        else:
            removed.append(name)
    if have_systemctl:
        subprocess.run(
            ["systemctl", "--user", "daemon-reload"],
            capture_output=True,
            check=False,
        )
    return removed


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