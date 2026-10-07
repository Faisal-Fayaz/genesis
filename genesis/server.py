"""The local gallery server.

Standard library only, on purpose. A wallpaper gallery is a single-user tool on
localhost; adding a web framework would mean a dependency to pin, audit, and
keep current for one page. `http.server` is enough, and it means the gallery
cannot break because of a version bump.

Binding is restricted to 127.0.0.1. The server can render and write to the
library, which is more capability than belongs on a network interface.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from . import library as library_module
from . import registry
from . import wallpaper as wallpaper_module
from .generators._common import to_image
from .recipe import PRESETS, Recipe

WEB_ROOT = Path(__file__).parent / "web"
PREVIEW_EDGE = 640

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".png": "image/png",
    ".json": "application/json; charset=utf-8",
}


@dataclass
class GalleryState:
    """Everything a request handler needs, opened once per request."""

    library: library_module.Library


class GalleryHandler(BaseHTTPRequestHandler):
    """Serves the gallery, its JSON, and its images."""

    server_version = "genesis"
    protocol_version = "HTTP/1.1"

    # -- plumbing ---------------------------------------------------------

    def log_message(self, format: str, *args: Any) -> None:
        """Quieter than the default, which logs every image request.

        Set GENESIS_GALLERY_DEBUG=1 to see requests and handler tracebacks.
        That is how the threading bug in the library was diagnosed without a
        debugger attached.
        """
        if os.environ.get("GENESIS_GALLERY_DEBUG"):
            sys.stderr.write(f"gallery: {format % args}\n")

    @property
    def library(self) -> library_module.Library:
        return self.server.library  # type: ignore[attr-defined]

    def _send(
        self, status: HTTPStatus, body: bytes, content_type: str
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json({"error": message}, status=status)

    def _file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self._error(HTTPStatus.NOT_FOUND, f"no such file: {path.name}")
            return
        content_type = CONTENT_TYPES.get(path.suffix, "application/octet-stream")
        self._send(HTTPStatus.OK, path.read_bytes(), content_type)

    # -- routing ----------------------------------------------------------

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = unquote(parsed.path)
        query = parse_qs(parsed.query)

        try:
            if route in ("/", "/index.html"):
                self._file(WEB_ROOT / "index.html")
            elif route == "/app.js":
                self._file(WEB_ROOT / "app.js")
            elif route == "/style.css":
                self._file(WEB_ROOT / "style.css")
            elif route == "/api/generators":
                self._json(registry_payload())
            elif route == "/api/post-schema":
                self._json(post_payload())
            elif route == "/api/palettes":
                self._json(PRESETS)
            elif route == "/api/pieces":
                self._json(self._list_pieces(query))
            elif route.startswith("/image/"):
                self._serve_entry_image(route.split("/", 2)[-1], "image")
            elif route.startswith("/preview/"):
                self._serve_entry_image(route.split("/", 2)[-1], "preview")
            elif route == "/api/wallpaper":
                self._json(wallpaper_payload())
            else:
                self._error(HTTPStatus.NOT_FOUND, f"no such route: {route}")
        except Exception as exc:  # noqa: BLE001
            # A handler that raises would otherwise drop the connection with no
            # response, leaving the page spinning. The specific cases above
            # already report their own status; this is the backstop, so it is
            # deliberately broad.
            if os.environ.get("GENESIS_GALLERY_DEBUG"):
                traceback.print_exc()
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        route = unquote(parsed.path)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._error(HTTPStatus.BAD_REQUEST, "request body is not valid JSON")
            return

        try:
            if route == "/api/render":
                self._render(payload)
            elif route == "/api/star":
                self._star(payload)
            elif route == "/api/apply":
                self._apply(payload)
            else:
                self._error(HTTPStatus.NOT_FOUND, f"no such route: {route}")
        except ValueError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except KeyError as exc:
            self._error(HTTPStatus.BAD_REQUEST, _clean(exc))
        except wallpaper_module.WallpaperUnavailable as exc:
            # Not a server fault: there is simply no session to change.
            self._error(HTTPStatus.CONFLICT, str(exc))
        except Exception as exc:  # noqa: BLE001
            # Backstop, as above: answer with a status rather than dropping the
            # connection. Broad on purpose.
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    # -- handlers ---------------------------------------------------------

    def _list_pieces(self, query: dict[str, list[str]]) -> dict[str, Any]:
        """List pieces, filtered.

        `count` is how many matched the filters and `total` is the size of the
        library, so the gallery can say "12 of 240" without a second request.
        """
        starred = query.get("starred", [None])[0]
        generator = query.get("generator", [None])[0]
        palette = query.get("palette", [None])[0]
        theme = query.get("theme", [None])[0]
        starred_filter = None if starred is None else starred in ("1", "true")

        entries = self.library.list(
            limit=int(query.get("limit", ["200"])[0]),
            generator=generator,
            palette=palette,
            theme=theme,
            starred=starred_filter,
        )
        return {
            "pieces": [entry.to_json() for entry in entries],
            "count": self.library.count(
                generator=generator, palette=palette, theme=theme, starred=starred_filter
            ),
            "total": self.library.count(),
            "starred_only": starred is not None,
        }

    def _serve_entry_image(self, digest: str, kind: str) -> None:
        entry = self.library.get(digest)
        if entry is None:
            self._error(HTTPStatus.NOT_FOUND, "unknown digest")
            return
        if kind == "preview" and entry.preview_path is not None:
            self._file(entry.preview_path)
            return
        self._file(entry.image_path)

    def _render(self, payload: dict[str, Any]) -> None:
        """Render a recipe from the posted JSON and record it.

        `preview` is a request flag rather than a recipe field, so it is read
        before validation: passing it through would make every request with a
        preview fail as an unknown field.
        """
        registry.ensure_loaded()
        payload = dict(payload)
        preview = bool(payload.pop("preview", False))
        recipe = Recipe.from_dict(payload)

        from . import render as render_module

        target = render_module.preview_size(recipe, PREVIEW_EDGE) if preview else recipe
        started = time.perf_counter()
        result = render_module.render(target)
        elapsed = time.perf_counter() - started

        entry = library_module.store(
            self.library, target, result, render_seconds=elapsed
        )
        self._json({"piece": entry.to_json(), "render_seconds": elapsed})

    def _star(self, payload: dict[str, Any]) -> None:
        digest = payload.get("digest")
        if not digest:
            self._error(HTTPStatus.BAD_REQUEST, "digest is required")
            return
        value = self.library.toggle_starred(digest)
        if value is None:
            self._error(HTTPStatus.NOT_FOUND, "unknown digest")
            return
        self._json({"digest": digest, "starred": value})

    def _apply(self, payload: dict[str, Any]) -> None:
        digest = payload.get("digest")
        if not digest:
            self._error(HTTPStatus.BAD_REQUEST, "digest is required")
            return
        entry = self.library.get(digest)
        if entry is None:
            self._error(HTTPStatus.NOT_FOUND, "unknown digest")
            return
        wallpaper_module.apply(entry.image_path, theme="both")
        self.library.record_applied(digest)
        self._json({"digest": digest, "applied": True})


def registry_payload() -> list[dict[str, Any]]:
    """Generator definitions with parameter metadata for the form builder."""
    registry.ensure_loaded()
    return [generator.to_json() for generator in registry.all_generators()]


def post_payload() -> dict[str, Any]:
    """Post-processing options, in the same shape as a generator's schema.

    Sent to the browser so the post-processing controls are generated from the
    same declarations that validate them.
    """
    from . import post as post_module

    return {"name": "post", "post": post_module.SCHEMA.to_json()}


def wallpaper_payload() -> dict[str, Any]:
    state = wallpaper_module.read_state()
    return {
        "dark": state.dark,
        "light": state.light,
        "session": wallpaper_module.has_session(),
        "display": wallpaper_module._display_size(),
    }


def _clean(exc: Exception) -> str:
    """Registry lookups raise KeyError, whose str() adds quotes."""
    message = str(exc)
    return message[1:-1] if message.startswith("'") and message.endswith("'") else message


class GalleryServer(ThreadingHTTPServer):
    """A threading server that owns the library."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], library: library_module.Library) -> None:
        super().__init__(address, GalleryHandler)
        self.library = library


def build_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    root: str | Path | None = None,
) -> GalleryServer:
    """Create a server, refusing anything but a loopback bind."""
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError(
            f"refusing to bind {host!r}: the gallery can write to the library "
            "and render, so it is restricted to localhost"
        )
    library = library_module.Library(root)
    library.open()
    return GalleryServer((host, port), library)


def serve_in_thread(
    host: str = "127.0.0.1", port: int = 0, root: str | Path | None = None
) -> tuple[GalleryServer, Any]:
    """Start a server on a background thread. Used by tests."""
    import threading

    server = build_server(host, port, root)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def to_png_bytes(array: Any) -> bytes:
    """Encode a linear-light array as PNG, for tests and exports."""
    import io

    buffer = io.BytesIO()
    to_image(array).save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()