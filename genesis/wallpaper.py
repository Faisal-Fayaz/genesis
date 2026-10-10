"""Setting the desktop wallpaper.

GNOME stores the wallpaper as a `file://` URI in gsettings, so applying one is
really two operations: put an image somewhere stable, then point the session at
it. The second part is where the care is needed.

A wallpaper path that does not exist renders as a black desktop with no
explanation. And GNOME keeps separate wallpapers for light and dark themes, so
setting only one leaves the other showing the previous image whenever the theme
flips.

The session check is not optional in the usual way. `gsettings` exits 0 while
doing nothing when there is no DBus session to reach, so a timer firing before
anyone logs in would appear to succeed and change nothing. Every path that writes
gsettings checks for a session first.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

SCHEMA = "org.gnome.desktop.background"
DARK_KEY = "picture-uri-dark"
LIGHT_KEY = "picture-uri"
DEFAULT_DESTINATION = Path.home() / ".local" / "share" / "genesis" / "wallpaper"


class WallpaperUnavailable(RuntimeError):
    """Raised when the running session cannot accept a wallpaper change.

    Carries a message meant for a person reading a log, since the usual trigger
    is a timer firing before anyone has logged in.
    """


@dataclass(frozen=True)
class AppliedState:
    """What the desktop is currently showing."""

    dark: str | None
    light: str | None

    @property
    def current(self) -> str | None:
        return self.dark or self.light


def has_session() -> bool:
    """Whether a graphical session exists that gsettings can reach.

    A systemd user timer can fire at login time, before the session's DBus is
    up. `gsettings` then exits 0 while doing nothing, so this is checked up front
    rather than trusting the exit status.

    The DBus address is required as well as a display: a display can be
    inherited from a parent environment while the bus belongs to a different
    session, which is the case a background timer actually hits.
    """
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False
    return bool(os.environ.get("DBUS_SESSION_BUS_ADDRESS"))


def require_session() -> None:
    if not has_session():
        raise WallpaperUnavailable(
            "no graphical session available (DISPLAY, WAYLAND_DISPLAY and "
            "DBUS_SESSION_BUS_ADDRESS are all unset); skipping wallpaper change"
        )


def _gsettings(*args: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["gsettings", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if check and completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise WallpaperUnavailable(f"gsettings {' '.join(args)} failed: {detail}")
    return completed.stdout.strip()


def read_state() -> AppliedState:
    """The URIs the desktop is currently showing."""
    def value(key: str) -> str | None:
        raw = _gsettings("get", SCHEMA, key, check=False)
        if not raw or raw == "''":
            return None
        return raw.strip("'")

    return AppliedState(dark=value(DARK_KEY), light=value(LIGHT_KEY))


def uri_for(path: Path) -> str:
    """A file:// URI for a wallpaper path."""
    return path.resolve().as_uri()


def stage(source: Path, destination: Path = DEFAULT_DESTINATION) -> Path:
    """Copy an image to a stable location.

    The staged file must not move: gsettings holds a URI, not a copy, so the
    original could be deleted out from under the desktop. The write goes via a
    temporary sibling and is renamed into place, so a reader never sees a
    half-written file and a failed render cannot blank the wallpaper.
    """
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"no such image: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.tmp")
    shutil.copyfile(source, staging)
    staging.replace(destination)
    return destination


def apply(
    source: Path,
    destination: Path = DEFAULT_DESTINATION,
    theme: str = "dark",
    stage_copy: bool = True,
) -> AppliedState:
    """Set the wallpaper, for one theme or both.

    `theme` names which variant to change. Passing both applies the same image to
    light and dark, which is right when the image is theme-neutral and avoids
    the desktop showing a stale image after a theme switch.
    """
    require_session()

    target = stage(source, destination) if stage_copy else Path(source).resolve()
    uri = uri_for(target)

    if theme == "both":
        keys: Sequence[str] = (DARK_KEY, LIGHT_KEY)
    elif theme == "light":
        keys = (LIGHT_KEY,)
    else:
        keys = (DARK_KEY,)

    for key in keys:
        _gsettings("set", SCHEMA, key, uri)
    return read_state()


DARK_DESTINATION = DEFAULT_DESTINATION.with_name("wallpaper-dark.png")
LIGHT_DESTINATION = DEFAULT_DESTINATION.with_name("wallpaper-light.png")


def apply_pair(
    dark_source: Path,
    light_source: Path,
    dark_destination: Path = DARK_DESTINATION,
    light_destination: Path = LIGHT_DESTINATION,
) -> AppliedState:
    """Set separate images for the light and dark themes.

    Derived from one recipe: the composition is shared and only the palette
    differs, so the two variants read as a pair rather than as two pictures.
    """
    require_session()
    dark = stage(dark_source, dark_destination)
    light = stage(light_source, light_destination)
    _gsettings("set", SCHEMA, DARK_KEY, uri_for(dark))
    _gsettings("set", SCHEMA, LIGHT_KEY, uri_for(light))
    return read_state()


def current_path(state: AppliedState | None = None) -> Path | None:
    """The local path the desktop is showing, if any."""
    state = state or read_state()
    uri = state.current
    if not uri or not uri.startswith("file://"):
        return None
    from urllib.parse import unquote, urlparse

    return Path(unquote(urlparse(uri).path))


def fits_display(image_size: tuple[int, int]) -> bool:
    """Whether an image matches the primary display.

    A wallpaper smaller than the screen is scaled up by GNOME, which softens
    fine structure. Larger is cropped, so it is not wrong, merely wasteful.
    """
    size = _display_size()
    if size is None:
        return True
    return image_size[0] >= size[0] and image_size[1] >= size[1]


def display_size() -> tuple[int, int] | None:
    """The primary display's resolution, or None if it cannot be determined.

    Public wrapper over the probing logic: the live engine sizes its output
    to the real display rather than assuming full HD.
    """
    return _display_size()


def _display_size() -> tuple[int, int] | None:
    """The primary display's resolution, or None if it cannot be determined."""
    for probe in (_display_size_xrandr, _display_size_gnome):
        size = probe()
        if size is not None:
            return size
    return None


def _display_size_xrandr() -> tuple[int, int] | None:
    if shutil.which("xrandr") is None:
        return None
    completed = subprocess.run(
        ["xrandr", "--current"], capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        return None
    for line in completed.stdout.splitlines():
        # The active output is marked with an asterisk after its mode.
        if " connected" in line and "*" in line:
            size = _parse_mode(line)
            if size is not None:
                return size
    return None


def _display_size_gnome() -> tuple[int, int] | None:
    """Fall back to Mutter's own idea of the screen size."""
    value = _gsettings("get", "org.gnome.desktop.interface", "dimensions", check=False)
    return _parse_mode(value)


def _parse_mode(text: str) -> tuple[int, int] | None:
    """Pull a WIDTHxHEIGHT resolution out of a line of display output.

    Display output is inconsistent: xrandr reports `2560x1440+0+0` with the
    origin attached, and Mutter reports `(1920, 1080)` with a comma. Both are
    handled, and an unparseable line yields None rather than a wrong answer,
    since the caller treats None as "assume it fits".
    """
    # Collapse whitespace first, so `"(1280, 1024)"` and `"1280x1080+0+0"` both
    # reach the scan as single well-formed tokens.
    candidate = " ".join(text.replace("(", " ").replace(")", " ").split())
    for token in candidate.split():
        for part in token.split("+"):
            width, separator, height = part.partition("x")
            if not separator:
                continue
            if width.isdigit() and height.isdigit():
                try:
                    return int(width), int(height)
                except ValueError:
                    continue

    # Comma-separated pairs, e.g. Mutter's `(1920, 1080)`.
    numbers = re.findall(r"\d+", candidate)
    if len(numbers) >= 2:
        return int(numbers[0]), int(numbers[1])
    return None