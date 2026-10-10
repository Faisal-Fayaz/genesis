"""The X11 backend: pixel format, failure modes, and a real server round trip.

Pixel conversion and error paths run headlessly. Window creation needs a
server, so the round trip starts its own Xvfb and skips cleanly where none is
available. Nothing here touches the developer's real display or desktop.
"""

from __future__ import annotations

import ctypes
import shutil
import socket
import subprocess
import time

import pytest

from genesis.live.backend import LiveBackendError, X11Backend, _to_server_pixels
from genesis.live.compositor import DirtyTile


def test_rgb_tile_expands_to_bgrx() -> None:
    tile = DirtyTile(x=0, y=0, width=2, height=1, data=bytes([10, 20, 30, 40, 50, 60]))
    expanded = _to_server_pixels(tile)
    assert len(expanded) == 8
    assert expanded[:4] == bytes([30, 20, 10, 0])
    assert expanded[4:] == bytes([60, 50, 40, 0])


def test_unreachable_display_is_a_clean_error(monkeypatch) -> None:
    monkeypatch.setenv("DISPLAY", ":99-no-such-server")
    with pytest.raises(LiveBackendError, match="cannot open X display"):
        X11Backend(64, 36, display_name=":99-no-such-server")


def _free_display() -> str:
    for number in range(90, 120):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", 6000 + number))
            return f":{number}"
        except OSError:
            continue
        finally:
            sock.close()
    raise RuntimeError("no free X display number")


def test_window_round_trip_on_xvfb(tmp_path) -> None:
    """Create, blit, query, and destroy a window on a throwaway server."""
    if shutil.which("Xvfb") is None:
        pytest.skip("Xvfb is not available")
    display = _free_display()
    server = subprocess.Popen(
        ["Xvfb", display, "-screen", "0", "640x360x24"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        backend = None
        for _ in range(100):
            try:
                backend = X11Backend(320, 180, display_name=display)
                break
            except LiveBackendError:
                time.sleep(0.05)
        assert backend is not None, "Xvfb never accepted the connection"
        try:
            assert backend.query_pointer() is None or isinstance(
                backend.query_pointer().x, float
            )
            assert backend.drain_events() in ("none", "redraw")
            backend.blit_full(bytes([7] * 320 * 180 * 3))
            backend.blit_full(bytes([9] * 320 * 180 * 3))
        finally:
            backend.close()
            backend.close()  # teardown is idempotent
    finally:
        server.terminate()
        server.wait(timeout=10)


def test_fullscreen_focus_probe_on_xvfb() -> None:
    """The focus probe reads real window state: focused, fullscreen, neither."""
    if shutil.which("Xvfb") is None:
        pytest.skip("Xvfb is not available")
    display = _free_display()
    server = subprocess.Popen(
        ["Xvfb", display, "-screen", "0", "640x360x24"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        backend = None
        for _ in range(100):
            try:
                backend = X11Backend(320, 180, display_name=display)
                break
            except LiveBackendError:
                time.sleep(0.05)
        assert backend is not None, "Xvfb never accepted the connection"
        try:
            x11, display_ptr = backend._x11, backend._display
            screen = x11.XDefaultScreen(display_ptr)
            root = x11.XRootWindow(display_ptr, screen)
            active_atom = x11.XInternAtom(display_ptr, b"_NET_ACTIVE_WINDOW", False)
            state_atom = x11.XInternAtom(display_ptr, b"_NET_WM_STATE", False)
            fs_atom = x11.XInternAtom(display_ptr, b"_NET_WM_STATE_FULLSCREEN", False)
            # Fresh server: nothing focused, nothing fullscreen.
            assert backend.query_fullscreen_focused() is False

            win = (ctypes.c_ulong * 1)(backend._window)
            x11.XChangeProperty(
                display_ptr, root, active_atom, 4, 32, 0, win, 1
            )
            state = (ctypes.c_ulong * 1)(fs_atom)
            x11.XChangeProperty(
                display_ptr, backend._window, state_atom, 4, 32, 0, state, 1
            )
            assert backend.query_fullscreen_focused() is True

            # Focused but plain: no yield.
            empty = (ctypes.c_ulong * 0)()
            x11.XChangeProperty(
                display_ptr, backend._window, state_atom, 4, 32, 0, empty, 0
            )
            assert backend.query_fullscreen_focused() is False
        finally:
            backend.close()
    finally:
        server.terminate()
        server.wait(timeout=10)
