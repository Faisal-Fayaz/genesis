"""Upload contracts for live frames.

`DirtyTile` is the unit the X11 backend converts: origin, size, and packed
RGB bytes, expanded to the server's BGRX layout by `_to_server_pixels`. The
fluid engine uploads whole frames, so tiles are always full-frame; the type
survives because the pixel-format contract is worth pinning down.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DirtyTile:
    """One tile to convert and upload: origin, size, and packed bytes."""

    x: int
    y: int
    width: int
    height: int
    data: bytes
