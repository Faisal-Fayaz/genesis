"""Applying a wallpaper to the desktop.

`gsettings` is the real system here, so every test runs against a stub that
records what would have been written. That is deliberate: the behaviours worth
pinning down are the ones where gsettings behaves surprisingly, and a test that
actually changed the desktop could not be run unattended.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from genesis import wallpaper as wallpaper_module


class FakeGsettings:
    """Records gsettings calls and replays a configured state."""

    def __init__(self, dark: str | None = None, light: str | None = None) -> None:
        self.values: dict[str, str] = {}
        self.calls: list[tuple[str, ...]] = []
        self.fail_on_set = False
        if dark:
            self.values["picture-uri-dark"] = dark
        if light:
            self.values["picture-uri"] = light

    def run(self, *args: str, check: bool = True) -> str:
        """Stand in for the `_gsettings` helper, which returns the stdout text."""
        del check
        self.calls.append(args)
        verb = args[0]
        if verb == "set":
            if self.fail_on_set:
                raise wallpaper_module.WallpaperUnavailable(
                    "gsettings set failed: no such schema"
                )
            self.values[args[2]] = args[3]
            return ""
        if verb == "get":
            stored = self.values.get(args[2], "")
            return "'" + stored + "'"
        raise wallpaper_module.WallpaperUnavailable(f"unsupported: {args}")

    @property
    def written(self) -> dict[str, str]:
        return {args[2]: args[3] for args in self.calls if args[0] == "set"}


@pytest.fixture
def fake(monkeypatch):
    stub = FakeGsettings()
    monkeypatch.setattr(wallpaper_module, "_gsettings", stub.run)
    monkeypatch.setattr(wallpaper_module, "has_session", lambda: True)
    return stub


@pytest.fixture
def image(tmp_path) -> Path:
    path = tmp_path / "wallpaper-source.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    return path


@pytest.fixture
def session(monkeypatch):
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/run/user/1000/bus")


class TestSessionDetection:
    def test_needs_a_display(self, monkeypatch) -> None:
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/x")
        assert wallpaper_module.has_session() is False

    def test_accepts_wayland(self, monkeypatch) -> None:
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/x")
        assert wallpaper_module.has_session() is True

    def test_needs_a_bus(self, monkeypatch) -> None:
        """The failure a timer hits: a display but no reachable session bus."""
        monkeypatch.setenv("DISPLAY", ":0")
        monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        assert wallpaper_module.has_session() is False

    def test_require_raises_without_a_session(self, monkeypatch) -> None:
        monkeypatch.setattr(wallpaper_module, "has_session", lambda: False)
        with pytest.raises(wallpaper_module.WallpaperUnavailable, match="no graphical session"):
            wallpaper_module.require_session()

    def test_a_background_timer_cannot_silently_fail(self, image, monkeypatch, tmp_path):
        """The whole point: no session means no gsettings call at all."""
        calls: list[tuple] = []
        monkeypatch.setattr(
            wallpaper_module,
            "_gsettings",
            lambda *a, **k: calls.append(a),
        )
        monkeypatch.setattr(wallpaper_module, "has_session", lambda: False)

        with pytest.raises(wallpaper_module.WallpaperUnavailable):
            wallpaper_module.apply(image, destination=tmp_path / "w.png")
        assert calls == []


class TestStage:
    def test_copies_the_image(self, image, tmp_path) -> None:
        target = wallpaper_module.stage(image, tmp_path / "wallpaper.png")
        assert target.read_bytes() == image.read_bytes()

    def test_creates_missing_directories(self, image, tmp_path) -> None:
        target = wallpaper_module.stage(image, tmp_path / "a" / "b" / "wallpaper.png")
        assert target.exists()

    def test_leaves_no_temporary_file(self, image, tmp_path) -> None:
        wallpaper_module.stage(image, tmp_path / "wallpaper.png")
        # The source lives in tmp_path too, so exclude it rather than requiring
        # a single entry.
        staged = [p.name for p in tmp_path.iterdir() if p.name != image.name]
        assert staged == ["wallpaper.png"]

    def test_overwrites_atomically(self, image, tmp_path) -> None:
        target = wallpaper_module.stage(image, tmp_path / "wallpaper.png")
        image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"1" * 64)
        wallpaper_module.stage(image, target)
        assert target.read_bytes() != b"\x89PNG\r\n\x1a\n" + b"0" * 64

    def test_missing_source_is_an_error(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            wallpaper_module.stage(tmp_path / "nope.png", tmp_path / "w.png")


class TestUri:
    def test_produces_a_file_uri(self) -> None:
        assert wallpaper_module.uri_for(Path("/tmp/x.png")) == "file:///tmp/x.png"

    def test_resolves_relative_paths(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "x.png").touch()
        assert wallpaper_module.uri_for(Path("x.png")).startswith("file://")

    def test_escapes_spaces(self, tmp_path) -> None:
        uri = wallpaper_module.uri_for(tmp_path / "my wallpaper.png")
        assert " " not in uri


class TestApply:
    def test_sets_the_dark_key_by_default(self, fake, image, tmp_path, session) -> None:
        wallpaper_module.apply(image, destination=tmp_path / "w.png")
        assert set(fake.written) == {"picture-uri-dark"}

    def test_sets_the_light_key(self, fake, image, tmp_path, session) -> None:
        wallpaper_module.apply(image, tmp_path / "w.png", theme="light")
        assert set(fake.written) == {"picture-uri"}

    def test_both_themes(self, fake, image, tmp_path, session) -> None:
        wallpaper_module.apply(image, tmp_path / "w.png", theme="both")
        assert set(fake.written) == {"picture-uri-dark", "picture-uri"}

    def test_both_themes_use_the_same_uri(self, fake, image, tmp_path, session) -> None:
        wallpaper_module.apply(image, tmp_path / "w.png", theme="both")
        written = fake.written
        assert written["picture-uri"] == written["picture-uri-dark"]

    def test_points_at_the_staged_copy(self, fake, image, tmp_path, session) -> None:
        wallpaper_module.apply(image, tmp_path / "w.png")
        staged = wallpaper_module.current_path(fake_state(fake))
        assert staged == (tmp_path / "w.png").resolve()

    def test_staged_copy_survives_the_original(self, fake, image, tmp_path, session) -> None:
        """The staged file must not depend on where the render was written."""
        wallpaper_module.apply(image, tmp_path / "w.png")
        image.unlink()
        assert (tmp_path / "w.png").exists()

    def test_no_stage_uses_the_original_path(self, fake, image, session) -> None:
        wallpaper_module.apply(image, stage_copy=False)
        assert wallpaper_module.current_path(fake_state(fake)) == image.resolve()

    def test_returns_the_new_state(self, fake, image, tmp_path, session) -> None:
        state = wallpaper_module.apply(image, tmp_path / "w.png", theme="both")
        assert state.dark is not None
        assert state.light is not None

    def test_a_failing_gsettings_call_is_reported(self, fake, image, tmp_path, session) -> None:
        fake.fail_on_set = True
        with pytest.raises(wallpaper_module.WallpaperUnavailable, match="gsettings"):
            wallpaper_module.apply(image, tmp_path / "w.png")


class TestApplyPair:
    def test_sets_distinct_images(self, fake, image, tmp_path, session) -> None:
        other = tmp_path / "light.png"
        other.write_bytes(image.read_bytes())
        state = wallpaper_module.apply_pair(
            image,
            other,
            dark_destination=tmp_path / "d.png",
            light_destination=tmp_path / "l.png",
        )
        assert state.dark != state.light
        assert state.dark.endswith("d.png")
        assert state.light.endswith("l.png")


def fake_state(stub: FakeGsettings) -> wallpaper_module.AppliedState:
    def value(key: str) -> str | None:
        raw = stub.values.get(key)
        return raw.strip("'") if raw else None

    return wallpaper_module.AppliedState(
        dark=value("picture-uri-dark"), light=value("picture-uri")
    )


class TestReadState:
    def test_reads_both_keys(self, monkeypatch) -> None:
        stub = FakeGsettings(dark="file:///a.png", light="file:///b.png")
        monkeypatch.setattr(wallpaper_module, "_gsettings", stub.run)
        state = wallpaper_module.read_state()
        assert state.dark == "file:///a.png"
        assert state.light == "file:///b.png"

    def test_none_when_unset(self, monkeypatch) -> None:
        stub = FakeGsettings()
        monkeypatch.setattr(wallpaper_module, "_gsettings", stub.run)
        state = wallpaper_module.read_state()
        assert state.dark is None
        assert state.current is None

    def test_current_prefers_dark(self) -> None:
        state = wallpaper_module.AppliedState(dark="file:///a", light="file:///b")
        assert state.current == "file:///a"

    def test_current_falls_back_to_light(self) -> None:
        state = wallpaper_module.AppliedState(dark=None, light="file:///b")
        assert state.current == "file:///b"

    def test_current_path_decodes_uris(self, monkeypatch) -> None:
        monkeypatch.setattr(
            wallpaper_module,
            "read_state",
            lambda: wallpaper_module.AppliedState(
                dark="file:///tmp/my%20image.png", light=None
            ),
        )
        assert wallpaper_module.current_path().name == "my image.png"

    def test_current_path_ignores_non_file_uris(self, monkeypatch) -> None:
        monkeypatch.setattr(
            wallpaper_module,
            "read_state",
            lambda: wallpaper_module.AppliedState(dark="https://x/y.png", light=None),
        )
        assert wallpaper_module.current_path() is None


class TestDisplaySize:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("1920x1080 144.00*+0+0", (1920, 1080)),
            ("1920, 1080", (1920, 1080)),
            ("2560x1440+0+0 (0x42)", (2560, 1440)),
            ("3840x2160+1920+0", (3840, 2160)),
            ("[('1920x1080')]", (1920, 1080)),
            ("3840 2160", (3840, 2160)),
            ("no resolution here", None),
            ("", None),
        ],
    )
    def test_parses_every_shape_a_display_reports(self, text, expected) -> None:
        assert wallpaper_module._parse_mode(text) == expected

    def test_falls_back_to_the_second_probe(self, monkeypatch) -> None:
        monkeypatch.setattr(wallpaper_module, "_display_size_xrandr", lambda: None)
        monkeypatch.setattr(
            wallpaper_module, "_display_size_gnome", lambda: (1280, 1024)
        )
        assert wallpaper_module._display_size() == (1280, 1024)

    def test_unknown_size_is_assumed_to_fit(self, monkeypatch) -> None:
        monkeypatch.setattr(wallpaper_module, "_display_size", lambda: None)
        assert wallpaper_module.fits_display((320, 240)) is True

    def test_too_small_is_reported(self, monkeypatch) -> None:
        monkeypatch.setattr(wallpaper_module, "_display_size", lambda: (1920, 1080))
        assert wallpaper_module.fits_display((800, 600)) is False

    def test_exactly_the_display_size_fits(self, monkeypatch) -> None:
        monkeypatch.setattr(wallpaper_module, "_display_size", lambda: (1920, 1080))
        assert wallpaper_module.fits_display((1920, 1080)) is True

    def test_larger_than_the_display_fits(self, monkeypatch) -> None:
        monkeypatch.setattr(wallpaper_module, "_display_size", lambda: (1920, 1080))
        assert wallpaper_module.fits_display((2560, 1440)) is True