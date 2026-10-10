"""Cursor-reactive live wallpapers at full resolution.

The still-image renderer in `genesis.render` is a one-shot pipeline: it builds
a complete noise field, traces hundreds of complete trails, and normalizes the
whole image. That costs seconds per frame and cannot drive a display.

The live engine here is a different execution model over the same mathematics:

- The angle field is built once per recipe and cached.
- A full-frame dye field starts pixel-identical to the still and is advected
  every frame, so every pixel moves rather than a few thousand particles.
- The cursor paints velocity into a decaying stir layer; ambient flow plus
  stir advect the dye, and a small per-frame relax pulls it back toward the
  still so hours of motion cannot homogenize the image.
- The simulation runs at half output resolution and upscales for display,
  keeping full-HD frames inside budget on a CPU.

Live output resembles the still renders but is interactive and therefore not
byte-identical to them. The still-image determinism guarantees are untouched:
the still renderer is called at most once per session, to acquire the base
layer, and never inside the frame loop.
"""

from __future__ import annotations
