"""Built-in generators.

Importing this package registers every generator with the registry. Generators
that are not yet written are simply absent from it, so the CLI and gallery
reflect whatever actually exists rather than a hardcoded list.
"""

from __future__ import annotations

from . import attractor, flowfield

__all__ = ["attractor", "flowfield"]
