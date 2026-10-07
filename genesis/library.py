"""The catalog of everything rendered.

A library is a SQLite database plus a directory of images. The database records
each piece's recipe, its digest, how long the render took, and whether it is
starred; the images live as ordinary PNGs on disk so they can be viewed, copied,
or set as a wallpaper without the catalog.

Two properties drive the design. First, records are keyed by the recipe's digest,
so rendering the same recipe twice is an update rather than a duplicate, and the
library cannot accumulate the same wallpaper twice. Second, the database is
derived state: it can be rebuilt from the recipes stored beside the images, so
losing it is recoverable and never the only copy of anything.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from . import render as render_module
from .recipe import Recipe
from .render import RenderResult

SCHEMA_VERSION = 1

# A piece, as the gallery and scheduler see it.
COLUMNS = (
    "digest",
    "slug",
    "generator",
    "width",
    "height",
    "palette_name",
    "theme",
    "recipe",
    "image_path",
    "preview_path",
    "render_seconds",
    "starred",
    "created_at",
    "applied_at",
    "applied_count",
)


def default_root() -> Path:
    """Where images and the catalog live, following the XDG base directory."""
    import os

    base = os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share")
    return Path(base) / "genesis"


def connect(path: Path) -> sqlite3.Connection:
    """Open a connection usable from any thread.

    The server is threaded, and SQLite connections are bound to the thread that
    created them. `check_same_thread=False` lifts that, which is safe here
    because every write is a single `execute` plus `commit`, and WAL gives
    readers their own snapshot without blocking.

    `busy_timeout` matters as much as the thread flag: without it a reader that
    arrives while another thread is committing gets `database is locked`
    immediately rather than waiting. On the thread-bound default it returned None
    from the query instead of raising, which surfaced as an unrelated error deep
    in a request handler.
    """
    connection = sqlite3.connect(path, check_same_thread=False, timeout=5.0)
    connection.row_factory = sqlite3.Row
    return connection


@dataclass(frozen=True)
class Entry:
    """One piece, read from the catalog at a point in time.

    An `Entry` is a snapshot, not a live view. Starring or applying a piece
    updates the row, not any `Entry` already handed out, so code that needs
    current values must re-read with `get` or `list`.
    """

    digest: str
    slug: str
    generator: str
    width: int
    height: int
    palette_name: str
    theme: str
    recipe: dict[str, Any]
    image_path: Path
    preview_path: Path | None
    render_seconds: float
    starred: bool
    created_at: float
    applied_at: float | None
    applied_count: int

    @property
    def recipe_object(self) -> Recipe:
        return Recipe.from_dict(self.recipe)

    @property
    def recipe_path(self) -> Path:
        """The recipe written beside the image, which makes this recoverable."""
        return self.image_path.with_suffix(".json")

    def to_json(self) -> dict[str, Any]:
        """A serializable form for the web gallery."""
        return {
            "digest": self.digest,
            "slug": self.slug,
            "generator": self.generator,
            "width": self.width,
            "height": self.height,
            "palette_name": self.palette_name,
            "theme": self.theme,
            "recipe": self.recipe,
            "image": f"/image/{self.digest}",
            "preview": f"/preview/{self.digest}" if self.preview_path else None,
            "render_seconds": round(self.render_seconds, 2),
            "starred": self.starred,
            "created_at": self.created_at,
            "applied_at": self.applied_at,
            "applied_count": self.applied_count,
        }


class Library:
    """A catalog of rendered pieces.

    Connections are per-thread. A single shared connection with
    `check_same_thread=False` is not enough: SQLite cursors are stateful, so two
    threads interleaving `execute` on one connection corrupt each other's
    results. It surfaced as a count query returning a bare tuple instead of a
    row, far from any obvious cause.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else default_root()
        self.images = self.root / "images"
        self.previews = self.root / "previews"
        self.database = self.root / "library.db"
        self._local = threading.local()
        self._shared: sqlite3.Connection | None = None

    # -- setup ------------------------------------------------------------

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def open(self) -> sqlite3.Connection:
        """Create the directory layout and return this thread's connection."""
        self.root.mkdir(parents=True, exist_ok=True)
        self.images.mkdir(parents=True, exist_ok=True)
        self.previews.mkdir(parents=True, exist_ok=True)

        connection = connect(self.database)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS pieces (
                digest         TEXT PRIMARY KEY,
                slug           TEXT NOT NULL,
                generator      TEXT NOT NULL,
                width          INTEGER NOT NULL,
                height         INTEGER NOT NULL,
                palette_name   TEXT NOT NULL,
                theme          TEXT NOT NULL,
                recipe         TEXT NOT NULL,
                image_path     TEXT NOT NULL,
                preview_path   TEXT,
                render_seconds REAL NOT NULL DEFAULT 0,
                starred        INTEGER NOT NULL DEFAULT 0,
                created_at     REAL NOT NULL,
                applied_at     REAL,
                applied_count  INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS pieces_created ON pieces(created_at DESC);
            CREATE INDEX IF NOT EXISTS pieces_generator ON pieces(generator);
            CREATE INDEX IF NOT EXISTS pieces_starred ON pieces(starred);

            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        connection.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        connection.commit()

        self._local.connection = connection
        # The connection opened here is also the one `close` will shut down. A
        # worker thread's own connection is closed when it finishes, since a
        # thread-local cannot be enumerated.
        self._shared = connection
        return connection

    def close(self) -> None:
        """Close this thread's connection.

        Connections belonging to finished worker threads are reclaimed by the
        garbage collector when their thread-local goes away.
        """
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            connection.close()
            del self._local.connection
        shared = self._shared
        if shared is not None and shared is not connection:
            shared.close()
        self._shared = None

    @property
    def connection(self) -> sqlite3.Connection:
        """This thread's connection, opening the catalog if necessary."""
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = self.open()
        return connection

    # -- writing ----------------------------------------------------------

    def add(
        self,
        recipe: Recipe,
        digest: str,
        image_path: Path,
        preview_path: Path | None = None,
        render_seconds: float = 0.0,
    ) -> Entry:
        """Record a render, replacing any earlier copy of the same recipe.

        Keying on the digest means re-rendering an unchanged recipe is a no-op
        rather than a duplicate row, and preserves the existing created_at and
        star.
        """
        entry = Entry(
            digest=digest,
            slug=recipe.slug(),
            generator=recipe.generator,
            width=recipe.width,
            height=recipe.height,
            palette_name=recipe.palette_name,
            theme=recipe.theme,
            recipe=recipe.to_dict(),
            image_path=image_path,
            preview_path=preview_path,
            render_seconds=render_seconds,
            starred=False,
            created_at=time.time(),
            applied_at=None,
            applied_count=0,
        )
        self.connection.execute(
            """
            INSERT INTO pieces (
                digest, slug, generator, width, height, palette_name, theme,
                recipe, image_path, preview_path, render_seconds, starred,
                created_at, applied_at, applied_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(digest) DO UPDATE SET
                recipe = excluded.recipe,
                image_path = excluded.image_path,
                preview_path = excluded.preview_path,
                render_seconds = excluded.render_seconds
            """,
            (
                entry.digest,
                entry.slug,
                entry.generator,
                entry.width,
                entry.height,
                entry.palette_name,
                entry.theme,
                json.dumps(entry.recipe, sort_keys=True),
                str(entry.image_path),
                str(entry.preview_path) if entry.preview_path else None,
                entry.render_seconds,
                int(entry.starred),
                entry.created_at,
                entry.applied_at,
                entry.applied_count,
            ),
        )
        self.connection.commit()
        return entry

    def set_starred(self, digest: str, starred: bool) -> bool:
        """Star or unstar a piece. Returns whether a row was affected."""
        cursor = self.connection.execute(
            "UPDATE pieces SET starred = ? WHERE digest = ?", (int(starred), digest)
        )
        self.connection.commit()
        return cursor.rowcount > 0

    def toggle_starred(self, digest: str) -> bool | None:
        """Flip a piece's star and return the new value, or None if unknown."""
        entry = self.get(digest)
        if entry is None:
            return None
        new_value = not entry.starred
        self.set_starred(digest, new_value)
        return new_value

    def record_applied(self, digest: str) -> None:
        """Note that a piece was set as the wallpaper."""
        self.connection.execute(
            """
            UPDATE pieces
            SET applied_at = ?, applied_count = applied_count + 1
            WHERE digest = ?
            """,
            (time.time(), digest),
        )
        self.connection.commit()

    def delete(self, digest: str, remove_files: bool = False) -> bool:
        """Remove a record, and optionally its images."""
        entry = self.get(digest)
        if entry is None:
            return False
        self.connection.execute("DELETE FROM pieces WHERE digest = ?", (digest,))
        self.connection.commit()
        if remove_files:
            for path in (entry.image_path, entry.preview_path):
                if path is not None:
                    Path(path).unlink(missing_ok=True)
        return True

    # -- reading ----------------------------------------------------------

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> Entry:
        return Entry(
            digest=row["digest"],
            slug=row["slug"],
            generator=row["generator"],
            width=row["width"],
            height=row["height"],
            palette_name=row["palette_name"],
            theme=row["theme"],
            recipe=json.loads(row["recipe"]),
            image_path=Path(row["image_path"]),
            preview_path=Path(row["preview_path"]) if row["preview_path"] else None,
            render_seconds=row["render_seconds"],
            starred=bool(row["starred"]),
            created_at=row["created_at"],
            applied_at=row["applied_at"],
            applied_count=row["applied_count"],
        )

    def get(self, digest: str) -> Entry | None:
        row = self.connection.execute(
            "SELECT * FROM pieces WHERE digest = ?", (digest,)
        ).fetchone()
        return self._row_to_entry(row) if row else None

    def find(self, slug: str) -> Entry | None:
        """Look up by slug, which is unique per recipe in practice."""
        row = self.connection.execute(
            "SELECT * FROM pieces WHERE slug = ? ORDER BY created_at DESC LIMIT 1",
            (slug,),
        ).fetchone()
        return self._row_to_entry(row) if row else None

    def latest(self) -> Entry | None:
        row = self.connection.execute(
            "SELECT * FROM pieces ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        return self._row_to_entry(row) if row else None

    def most_recently_applied(self) -> Entry | None:
        row = self.connection.execute(
            "SELECT * FROM pieces WHERE applied_at IS NOT NULL "
            "ORDER BY applied_at DESC LIMIT 1"
        ).fetchone()
        return self._row_to_entry(row) if row else None

    def list(
        self,
        limit: int = 100,
        offset: int = 0,
        generator: str | None = None,
        starred: bool | None = None,
        palette: str | None = None,
        theme: str | None = None,
    ) -> list[Entry]:
        clauses: list[str] = []
        params: list[Any] = []
        if generator:
            clauses.append("generator = ?")
            params.append(generator)
        if starred is not None:
            clauses.append("starred = ?")
            params.append(int(starred))
        if palette:
            clauses.append("palette_name = ?")
            params.append(palette)
        if theme:
            clauses.append("theme = ?")
            params.append(theme)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend([limit, offset])
        rows = self.connection.execute(
            f"SELECT * FROM pieces {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params,
        ).fetchall()
        return [self._row_to_entry(row) for row in rows]

    def count(self, **filters: Any) -> int:
        clauses: list[str] = []
        params: list[Any] = []
        for key, column in (
            ("generator", "generator"),
            ("palette", "palette_name"),
            ("theme", "theme"),
        ):
            if filters.get(key):
                clauses.append(f"{column} = ?")
                params.append(filters[key])
        if filters.get("starred") is not None:
            clauses.append("starred = ?")
            params.append(int(filters["starred"]))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        row = self.connection.execute(
            f"SELECT COUNT(*) AS n FROM pieces {where}", params
        ).fetchone()
        return int(row["n"]) if row is not None else 0

    def generators(self) -> list[str]:
        rows = self.connection.execute(
            "SELECT DISTINCT generator FROM pieces ORDER BY generator"
        ).fetchall()
        return [row["generator"] for row in rows]

    def digests(self) -> list[str]:
        rows = self.connection.execute(
            "SELECT digest FROM pieces ORDER BY created_at DESC"
        ).fetchall()
        return [row["digest"] for row in rows]

    def iter_all(self) -> Iterator[Entry]:
        rows = self.connection.execute("SELECT * FROM pieces ORDER BY created_at").fetchall()
        for row in rows:
            yield self._row_to_entry(row)

    # -- housekeeping -----------------------------------------------------

    def missing_images(self) -> list[Entry]:
        """Records whose image file is gone.

        The database is derived state, so a mismatch means an image was removed
        by hand. Reporting it lets the gallery show what is actually on disk.
        """
        return [entry for entry in self.iter_all() if not entry.image_path.exists()]

    def prune_orphans(self, remove_files: bool = False) -> int:
        """Drop records whose images are missing. Returns how many were removed."""
        orphans = self.missing_images()
        for entry in orphans:
            self.delete(entry.digest, remove_files=remove_files)
        return len(orphans)

    def rebuild(self) -> int:
        """Re-derive records from the recipes stored beside the images.

        The catalog can always be reconstructed, because each image has its
        recipe written next to it. This is the recovery path after the database
        is lost or the images are moved in from elsewhere.
        """
        self.connection.execute("DELETE FROM pieces")
        self.connection.commit()

        recovered = 0
        for recipe_path in sorted(self.images.glob("*.json")):
            image_path = recipe_path.with_suffix(".png")
            if not image_path.exists():
                continue
            try:
                recipe = Recipe.load(recipe_path)
                digest = render_module.render(recipe).digest
            except (ValueError, OSError, KeyError):
                continue
            self.add(recipe, digest, image_path)
            recovered += 1
        return recovered


def store(
    library: Library,
    recipe: Recipe,
    result: RenderResult,
    render_seconds: float = 0.0,
    with_preview: bool = True,
    preview_edge: int = 480,
) -> Entry:
    """Copy a rendered result into the library and record it."""
    image_path = library.images / f"{result.digest[:16]}-{recipe.slug()}.png"
    result.image.save(image_path, format="PNG", optimize=True)
    recipe.save(image_path.with_suffix(".json"))

    preview_path: Path | None = None
    if with_preview:
        preview = render_module.preview_size(recipe, preview_edge)
        if preview.width != recipe.width or preview.height != recipe.height:
            # Rendered from the same recipe, so the preview is a smaller copy of
            # exactly this image rather than a separately-sampled approximation.
            render_module.render_to_file(
                preview, library.previews / f"{result.digest[:16]}.png"
            )
            preview_path = library.previews / f"{result.digest[:16]}.png"

    return library.add(
        recipe,
        result.digest,
        image_path,
        preview_path,
        render_seconds,
    )


@contextmanager
def library_at(root: str | Path | None) -> Iterator[Library]:
    """Open a library for the duration of a block."""
    library = Library(root)
    library.open()
    try:
        yield library
    finally:
        library.close()


def parse_filter(value: str | None) -> tuple[bool | None, str | None]:
    """Parse a ``name=value`` filter argument."""
    if not value or "=" not in value:
        return None, None
    key, _, raw = value.partition("=")
    if key == "starred":
        return raw.lower() in ("1", "true", "yes"), None
    return None, raw


def describe_entry(entry: Entry) -> str:
    """A single-line human summary."""
    star = "*" if entry.starred else " "
    return (
        f"{star} {entry.digest[:12]}  {entry.generator:10s} "
        f"{entry.width}x{entry.height}  {entry.palette_name:6s}  "
        f"{datetime.fromtimestamp(entry.created_at, tz=UTC):%Y-%m-%d}"
    )


def sort_entries(entries: Sequence[Entry], key: str) -> list[Entry]:
    """Order a listing. `key` is a field name or 'recent'/'applied'/'starred'."""
    if key == "starred":
        return sorted(entries, key=lambda e: (not e.starred, -e.created_at))
    if key == "applied":
        return sorted(entries, key=lambda e: -(e.applied_at or 0.0))
    if key == "rendered":
        return sorted(entries, key=lambda e: -e.render_seconds)
    return sorted(entries, key=lambda e: -e.created_at)


def elapsed(seconds: float) -> str:
    """A short duration, for CLI output."""
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    return f"{seconds / 60:.1f}m"