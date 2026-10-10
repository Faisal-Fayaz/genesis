"""X11 presentation backend for live wallpapers.

Implemented with `ctypes` against libX11 and libXext directly, so the live
mode needs no new Python dependency. `python-xlib` would work too, but it does
not map MIT-SHM, and shared-memory upload is what keeps full-frame blits
inside the frame budget.

Frames go out whole. Shared memory is preferred: one full-frame SHM image is
attached once and each frame is memcpy'd into it and pushed with
`XShmPutImage`. Without SHM the frame goes out with one `XPutImage` from a
persistent staging buffer: a full 6 MB transfer per frame, visibly slower,
kept so live mode still runs where SHM is unavailable. (A dirty-tile upload
path existed in the particle era; the fluid engine's dye moves everywhere, so
tiles saved nothing and the path was removed.)

The window is a normal managed window typed `_NET_WM_WINDOW_TYPE_DESKTOP` and
lowered beneath application windows, with below/taskbar-skipping state hints.
This is X11-only by design: Wayland has no equivalent mechanism, and the lock
screen is a separate surface.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from dataclasses import dataclass
from typing import Self

from .compositor import DirtyTile

# Event types and masks from X11/X.h.
EXPOSE = 12
CONFIGURE_NOTIFY = 22
DESTROY_NOTIFY = 17
EXPOSURE_MASK = 1 << 15
STRUCTURE_NOTIFY_MASK = 1 << 17
ZPIXMAP = 2
PROP_MODE_REPLACE = 0
XA_ATOM = 4
IPC_PRIVATE = 0
IPC_CREAT = 0o1000
IPC_RMID = 0
SHM_RW = 0o600


class LiveBackendError(RuntimeError):
    """The display cannot host a live wallpaper."""


class _XErrorEvent(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("display", ctypes.c_void_p),
        ("resourceid", ctypes.c_ulong),
        ("serial", ctypes.c_ulong),
        ("error_code", ctypes.c_ubyte),
        ("request_code", ctypes.c_ubyte),
        ("minor_code", ctypes.c_ubyte),
    ]


_error_handler_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)


def _to_server_pixels(tile: DirtyTile) -> bytes:
    """Expand an RGB tile to the server's 32-bit little-endian BGRX layout.

    Depth-24 TrueColor visuals store each pixel in four bytes. On little-endian
    servers with the standard masks this is blue, green, red, pad. Anything
    else raises, and the caller falls back to refusing the visual rather than
    showing swapped channels.
    """
    import numpy as np

    pixels = np.frombuffer(tile.data, dtype=np.uint8).reshape(tile.height, tile.width, 3)
    padded = np.zeros((tile.height, tile.width, 4), dtype=np.uint8)
    padded[..., 0] = pixels[..., 2]
    padded[..., 1] = pixels[..., 1]
    padded[..., 2] = pixels[..., 0]
    return padded.tobytes()


class _XImage(ctypes.Structure):
    # Only the leading fields are accessed; function pointers at the tail are
    # never called through this definition.
    _fields_ = [
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("xoffset", ctypes.c_int),
        ("format", ctypes.c_int),
        ("data", ctypes.c_void_p),
        ("byte_order", ctypes.c_int),
        ("bitmap_unit", ctypes.c_int),
        ("bitmap_bit_order", ctypes.c_int),
        ("bitmap_pad", ctypes.c_int),
        ("depth", ctypes.c_int),
        ("bytes_per_line", ctypes.c_int),
        ("bits_per_pixel", ctypes.c_int),
        ("red_mask", ctypes.c_ulong),
        ("green_mask", ctypes.c_ulong),
        ("blue_mask", ctypes.c_ulong),
        ("obdata", ctypes.c_void_p),
    ]


class _ShmInfo(ctypes.Structure):
    _fields_ = [
        ("shmseg", ctypes.c_ulong),
        ("shmid", ctypes.c_int),
        ("shmaddr", ctypes.c_void_p),
        ("readOnly", ctypes.c_int),
    ]


def _load(name: str) -> ctypes.CDLL:
    path = ctypes.util.find_library(name)
    if path is None:
        raise LiveBackendError(f"system library {name!r} was not found")
    return ctypes.CDLL(path, use_errno=True)


@dataclass(frozen=True)
class PointerState:
    """Pointer position in root coordinates, or None when unknown."""

    x: float
    y: float


class X11Backend:
    """A fullscreen desktop-type window fed with whole frames."""

    def __init__(self, width: int, height: int, display_name: str | None = None) -> None:
        self.width = width
        self.height = height
        self._x11 = _load("X11")
        self._xext = _load("Xext")
        self._libc = _load("c")
        self._wire()
        self._display = self._x11.XOpenDisplay(display_name.encode() if display_name else None)
        if not self._display:
            raise LiveBackendError(
                f"cannot open X display {display_name or os.environ.get('DISPLAY')!r}"
            )
        self._x_error: tuple[int, int, int] | None = None

        @_error_handler_type
        def _record_error(display: object, event: object) -> int:
            details = ctypes.cast(event, ctypes.POINTER(_XErrorEvent)).contents
            self._x_error = (
                details.error_code, details.request_code, details.minor_code
            )
            return 0

        # Trap X errors instead of dying on them: a rejected SHM request or a
        # closed window becomes a fallback or a clean shutdown, not a crash.
        self._error_callback = _record_error
        self._x11.XSetErrorHandler(self._error_callback)
        screen = self._x11.XDefaultScreen(self._display)
        root = self._x11.XRootWindow(self._display, screen)
        black = self._x11.XBlackPixel(self._display, screen)
        depth = self._x11.XDefaultDepth(self._display, screen)
        if depth != 24:
            self._x11.XCloseDisplay(self._display)
            raise LiveBackendError(
                f"live wallpapers need a 24-bit TrueColor visual, found depth {depth}"
            )
        self._window = self._x11.XCreateSimpleWindow(
            self._display, root, 0, 0, width, height, 0, black, black
        )
        self._set_desktop_type()
        self._x11.XSelectInput(self._display, self._window, EXPOSURE_MASK | STRUCTURE_NOTIFY_MASK)
        self._gc = self._x11.XCreateGC(self._display, self._window, 0, None)
        self._x11.XMapWindow(self._display, self._window)
        self._x11.XLowerWindow(self._display, self._window)
        self._x11.XSync(self._display, False)
        self._shm_image = None
        self._shm_info = None
        self._shm_buffer = None
        self._shm_stride = 0
        self._use_shm = bool(self._try_attach_shm(screen))
        # The last server rejection, if shared memory was refused after
        # attaching. Recorded rather than raised so `--stats` can report why a
        # session runs the slower plain-transfer path.
        self.last_rejected_error: tuple[int, int, int] | None = None
        self._staging = bytearray(width * height * 4)
        # A persistent staging image for the non-SHM path: one XImage for the
        # session, filled tile by tile. It is deliberately never destroyed.
        # XDestroyImage segfaults on images with client-owned data on this
        # libX11, so the struct is left for process teardown to reclaim.
        self._staging_view = (ctypes.c_char * len(self._staging)).from_buffer(
            self._staging
        )
        self._staging_image = self._x11.XCreateImage(
            self._display,
            self._x11.XDefaultVisual(self._display, screen),
            24,
            ZPIXMAP,
            0,
            ctypes.cast(self._staging_view, ctypes.c_void_p),
            width,
            height,
            32,
            0,
        )
        if not self._staging_image:
            raise LiveBackendError("XCreateImage failed")
        self._closed = False

    def close(self) -> None:
        """Destroy the window and detach shared memory.

        Neither XImage is destroyed: XDestroyImage segfaults on images with
        client-owned or shared-memory data on this libX11, and both structs are
        tiny one-per-session allocations the OS reclaims on exit.
        """
        if self._closed:
            return
        self._closed = True
        try:
            if self._shm_image is not None:
                self._xext.XShmDetach(self._display, self._shm_info)
            if self._shm_buffer is not None:
                self._libc.shmdt(self._shm_buffer)
            if self._shm_info is not None and self._shm_info.shmid >= 0:
                self._libc.shmctl(self._shm_info.shmid, IPC_RMID, None)
            self._x11.XFreeGC(self._display, self._gc)
            self._x11.XDestroyWindow(self._display, self._window)
            self._x11.XSync(self._display, False)
            self._x11.XCloseDisplay(self._display)
        except Exception:  # noqa: BLE001, S110 - teardown must never raise
            pass

    def __enter__(self) -> Self:
        return self

    @property
    def using_shm(self) -> bool:
        """Whether uploads go through shared memory."""
        return self._use_shm and self._shm_image is not None

    def __exit__(self, *exc: object) -> None:
        self.close()

    def query_pointer(self) -> PointerState | None:
        """Current pointer position in root coordinates."""
        root = ctypes.c_ulong()
        child = ctypes.c_ulong()
        root_x = ctypes.c_int()
        root_y = ctypes.c_int()
        win_x = ctypes.c_int()
        win_y = ctypes.c_int()
        mask = ctypes.c_uint()
        screen = self._x11.XDefaultScreen(self._display)
        root_win = self._x11.XRootWindow(self._display, screen)
        ok = self._x11.XQueryPointer(
            self._display,
            root_win,
            ctypes.byref(root),
            ctypes.byref(child),
            ctypes.byref(root_x),
            ctypes.byref(root_y),
            ctypes.byref(win_x),
            ctypes.byref(win_y),
            ctypes.byref(mask),
        )
        if not ok:
            return None
        return PointerState(x=float(root_x.value), y=float(root_y.value))

    def drain_events(self) -> str:
        """Consume pending events: `quit`, `resized`, `redraw`, or `none`.

        A resize wins over a redraw (the next frame re-fits anyway); quit
        wins over everything.
        """
        result = "none"
        event = ctypes.create_string_buffer(192)
        while self._x11.XPending(self._display):
            self._x11.XNextEvent(self._display, event)
            kind = ctypes.c_int.from_buffer(event).value
            if kind == DESTROY_NOTIFY:
                return "quit"
            if kind == CONFIGURE_NOTIFY:
                result = "resized"
            elif kind == EXPOSE and result == "none":
                result = "redraw"
        return result

    def query_fullscreen_focused(self) -> bool | None:
        """Whether the focused window is fullscreen; None when indeterminable.

        Polled on a seconds cadence, never per frame: two round trips per
        call is nothing at 0.5 Hz and unacceptable at 60. Any X failure
        means unknown, and unknown means carry on — a sensor must never
        pause a healthy session.
        """
        try:
            screen = self._x11.XDefaultScreen(self._display)
            root = self._x11.XRootWindow(self._display, screen)
            active_atom = self._x11.XInternAtom(
                self._display, b"_NET_ACTIVE_WINDOW", False
            )
            focused = self._atom_list(root, active_atom)
            if not focused:
                return False
            state_atom = self._x11.XInternAtom(self._display, b"_NET_WM_STATE", False)
            states = self._atom_list(focused[0], state_atom)
            if states is None:
                return None
            fullscreen = self._x11.XInternAtom(
                self._display, b"_NET_WM_STATE_FULLSCREEN", False
            )
            return fullscreen in states
        except Exception:  # noqa: BLE001 - any X failure means "unknown"
            return None

    def _atom_list(self, window: int, atom: int) -> list[int] | None:
        """An ATOM property as Python ints, or None when absent/unreadable."""
        actual_type = ctypes.c_ulong()
        actual_format = ctypes.c_int()
        nitems = ctypes.c_ulong()
        bytes_after = ctypes.c_ulong()
        prop = ctypes.c_void_p()
        status = self._x11.XGetWindowProperty(
            self._display,
            window,
            atom,
            0,
            1024,
            False,
            XA_ATOM,
            ctypes.byref(actual_type),
            ctypes.byref(actual_format),
            ctypes.byref(nitems),
            ctypes.byref(bytes_after),
            ctypes.byref(prop),
        )
        if status != 0 or not prop:
            return None
        try:
            if actual_format.value != 32:
                return None
            array = ctypes.cast(prop, ctypes.POINTER(ctypes.c_ulong))
            return [array[i] for i in range(nitems.value)]
        finally:
            self._x11.XFree(prop)

    def _sync_check(self) -> tuple[int, int, int] | None:
        """Sync with the server and return any trapped X error, clearing it."""
        self._x11.XSync(self._display, False)
        error, self._x_error = self._x_error, None
        return error

    def _drop_shm(self) -> None:
        """Detach shared memory after the server rejects it.

        The SHM XImage struct is intentionally leaked here for the same reason
        as in `close`: destroying it segfaults on this libX11.
        """
        try:
            if self._shm_image is not None:
                self._xext.XShmDetach(self._display, self._shm_info)
            if self._shm_buffer is not None:
                self._libc.shmdt(self._shm_buffer)
            if self._shm_info is not None and self._shm_info.shmid >= 0:
                self._libc.shmctl(self._shm_info.shmid, IPC_RMID, None)
        except Exception:  # noqa: BLE001, S110 - already falling back; keep going
            pass
        self._shm_image = None
        self._shm_info = None
        self._shm_buffer = None
        self._use_shm = False

    def blit_full(self, data: bytes) -> None:
        """Upload a complete frame, via SHM when attached, else one blit."""
        expanded = _to_server_pixels(
            DirtyTile(x=0, y=0, width=self.width, height=self.height, data=data)
        )
        if self._use_shm and self._shm_image is not None:
            ctypes.memmove(self._shm_buffer, expanded, len(expanded))
            self._xext.XShmPutImage(
                self._display,
                self._window,
                self._gc,
                self._shm_image,
                0,
                0,
                0,
                0,
                self.width,
                self.height,
                False,
            )
            error = self._sync_check()
            if error is None:
                return
            self.last_rejected_error = error
            self._drop_shm()
        self._staging[:] = expanded
        self._x11.XPutImage(
            self._display,
            self._window,
            self._gc,
            self._staging_image,
            0,
            0,
            0,
            0,
            self.width,
            self.height,
        )
        self._x11.XSync(self._display, False)

    def _set_desktop_type(self) -> None:
        x11 = self._x11
        type_atom = x11.XInternAtom(self._display, b"_NET_WM_WINDOW_TYPE", False)
        desktop = x11.XInternAtom(self._display, b"_NET_WM_WINDOW_TYPE_DESKTOP", False)
        value = (ctypes.c_ulong * 1)(desktop)
        x11.XChangeProperty(
            self._display, self._window, type_atom, XA_ATOM, 32,
            PROP_MODE_REPLACE, value, 1,
        )
        state_atom = x11.XInternAtom(self._display, b"_NET_WM_STATE", False)
        below = x11.XInternAtom(self._display, b"_NET_WM_STATE_BELOW", False)
        skip_taskbar = x11.XInternAtom(self._display, b"_NET_WM_STATE_SKIP_TASKBAR", False)
        skip_pager = x11.XInternAtom(self._display, b"_NET_WM_STATE_SKIP_PAGER", False)
        states = (ctypes.c_ulong * 3)(below, skip_taskbar, skip_pager)
        x11.XChangeProperty(
            self._display, self._window, state_atom, XA_ATOM, 32,
            PROP_MODE_REPLACE, states, 3,
        )
        x11.XStoreName(self._display, self._window, b"Genesis Live Wallpaper")

    def _try_attach_shm(self, screen: int) -> bool:
        try:
            if not self._xext.XShmQueryExtension(self._display):
                return False
            visual = self._x11.XDefaultVisual(self._display, screen)
            depth = self._x11.XDefaultDepth(self._display, screen)
            if depth != 24:
                return False
            size = self.width * self.height * 4
            shmid = self._libc.shmget(IPC_PRIVATE, size, IPC_CREAT | SHM_RW)
            if shmid < 0:
                return False
            addr = self._libc.shmat(shmid, None, 0)
            if addr == ctypes.c_void_p(-1).value:
                self._libc.shmctl(shmid, IPC_RMID, None)
                return False
            info = _ShmInfo()
            info.shmid = shmid
            info.shmaddr = addr
            info.readOnly = False
            image = self._xext.XShmCreateImage(
                self._display, visual, depth, ZPIXMAP, None,
                ctypes.byref(info), self.width, self.height,
            )
            if not image:
                self._libc.shmdt(addr)
                self._libc.shmctl(shmid, IPC_RMID, None)
                return False
            if not self._xext.XShmAttach(self._display, ctypes.byref(info)):
                self._x11.XDestroyImage(image)
                self._libc.shmdt(addr)
                self._libc.shmctl(shmid, IPC_RMID, None)
                return False
            self._x11.XSync(self._display, False)
            self._shm_image = image
            self._shm_info = info
            self._shm_buffer = addr
            self._shm_stride = image.contents.bytes_per_line
            return True
        except Exception:  # noqa: BLE001 - any SHM failure means plain fallback
            return False

    def _wire(self) -> None:
        x11 = self._x11
        x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        x11.XOpenDisplay.restype = ctypes.c_void_p
        x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
        x11.XDefaultScreen.argtypes = [ctypes.c_void_p]
        x11.XDefaultScreen.restype = ctypes.c_int
        x11.XRootWindow.argtypes = [ctypes.c_void_p, ctypes.c_int]
        x11.XRootWindow.restype = ctypes.c_ulong
        x11.XBlackPixel.argtypes = [ctypes.c_void_p, ctypes.c_int]
        x11.XBlackPixel.restype = ctypes.c_ulong
        x11.XDefaultVisual.argtypes = [ctypes.c_void_p, ctypes.c_int]
        x11.XDefaultVisual.restype = ctypes.c_void_p
        x11.XDefaultDepth.argtypes = [ctypes.c_void_p, ctypes.c_int]
        x11.XDefaultDepth.restype = ctypes.c_int
        x11.XCreateSimpleWindow.argtypes = [
            ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_int,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_ulong, ctypes.c_ulong,
        ]
        x11.XCreateSimpleWindow.restype = ctypes.c_ulong
        x11.XSelectInput.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_long]
        x11.XCreateGC.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_void_p]
        x11.XCreateGC.restype = ctypes.c_void_p
        x11.XMapWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        x11.XLowerWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        x11.XSync.argtypes = [ctypes.c_void_p, ctypes.c_int]
        x11.XInternAtom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        x11.XInternAtom.restype = ctypes.c_ulong
        x11.XChangeProperty.argtypes = [
            ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong,
            ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
        ]
        x11.XStoreName.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_char_p]
        x11.XSetErrorHandler.argtypes = [ctypes.c_void_p]
        x11.XSetErrorHandler.restype = ctypes.c_void_p
        x11.XQueryPointer.argtypes = [
            ctypes.c_void_p, ctypes.c_ulong,
            ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_ulong),
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_uint),
        ]
        x11.XQueryPointer.restype = ctypes.c_int
        x11.XGetWindowProperty.argtypes = [
            ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_long,
            ctypes.c_long, ctypes.c_int, ctypes.c_ulong,
            ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_ulong),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        x11.XGetWindowProperty.restype = ctypes.c_int
        x11.XFree.argtypes = [ctypes.c_void_p]
        x11.XPending.argtypes = [ctypes.c_void_p]
        x11.XPending.restype = ctypes.c_int
        x11.XNextEvent.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        x11.XPutImage.argtypes = [
            ctypes.c_void_p, ctypes.c_ulong, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_uint, ctypes.c_uint,
        ]
        x11.XCreateImage.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_int, ctypes.c_int,
        ]
        x11.XCreateImage.restype = ctypes.POINTER(_XImage)
        x11.XDestroyImage.argtypes = [ctypes.POINTER(_XImage)]
        x11.XDestroyWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        x11.XFreeGC.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        xext = self._xext
        xext.XShmQueryExtension.argtypes = [ctypes.c_void_p]
        xext.XShmQueryExtension.restype = ctypes.c_int
        xext.XShmCreateImage.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_int,
            ctypes.c_void_p, ctypes.POINTER(_ShmInfo), ctypes.c_uint, ctypes.c_uint,
        ]
        xext.XShmCreateImage.restype = ctypes.POINTER(_XImage)
        xext.XShmAttach.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ShmInfo)]
        xext.XShmAttach.restype = ctypes.c_int
        xext.XShmDetach.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ShmInfo)]
        xext.XShmPutImage.argtypes = [
            ctypes.c_void_p, ctypes.c_ulong, ctypes.c_void_p, ctypes.POINTER(_XImage),
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_int,
        ]
        libc = self._libc
        libc.shmget.argtypes = [ctypes.c_int, ctypes.c_size_t, ctypes.c_int]
        libc.shmget.restype = ctypes.c_int
        libc.shmat.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
        libc.shmat.restype = ctypes.c_void_p
        libc.shmdt.argtypes = [ctypes.c_void_p]
        libc.shmctl.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
