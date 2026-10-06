"""Command-line interface.

Four verbs carry the workflow: `gen` renders a recipe, `gen random` invents one
when you have no parameters in mind, `list` shows what exists, and `describe`
shows how a recipe resolves. Overrides are layered onto a loaded recipe and the
result is revalidated, so an override that would produce an invalid render fails
immediately with a useful message.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import __version__, registry
from . import render as render_module
from .params import ParamSpec
from .recipe import PRESETS, Recipe, random_recipe
from .rng import random_seed, stream

DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
PREVIEW_EDGE = 640


def _error(message: str) -> int:
    print(f"genesis: {message}", file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="genesis",
        description="Deterministic procedural wallpaper engine",
    )
    parser.add_argument("--version", action="version", version=f"genesis {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("gen", help="render a recipe from a file")
    gen.add_argument(
        "recipe",
        nargs="?",
        type=Path,
        help="path to a recipe JSON file, or 'random' to invent one",
    )
    gen.add_argument("-o", "--output", type=Path, help="output PNG path")
    gen.add_argument(
        "--preview", action="store_true", help="render a small preview, not full size"
    )
    gen.add_argument(
        "--preview-edge", type=int, default=PREVIEW_EDGE, help="longest edge for --preview"
    )
    gen.add_argument(
        "--scale", type=float, default=None, help="multiply the recipe's dimensions"
    )
    gen.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a recipe field, e.g. --set palette_name=solar",
    )
    gen.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a generator parameter, e.g. --param trails=8000",
    )
    gen.add_argument(
        "--post",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a post option, e.g. --post grain=0.04",
    )
    gen.add_argument("--json", action="store_true", help="print the result as JSON")
    gen.add_argument(
        "--save", action="store_true", help="write the overridden recipe back to disk"
    )
    gen.add_argument("-q", "--quiet", action="store_true", help="suppress progress output")

    gen_random = sub.add_parser("gen-random", help="invent a recipe and render it")
    gen_random.add_argument("generator", nargs="?", help="generator to use (default: any)")
    gen_random.add_argument("--seed", type=int, help="pin the entropy seed")
    gen_random.add_argument("--palette", choices=sorted(PRESETS), help="palette preset")
    gen_random.add_argument("--light", action="store_true", help="light theme")
    gen_random.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    gen_random.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    gen_random.add_argument(
        "--preview", action="store_true", help="render a small preview instead"
    )
    gen_random.add_argument("-o", "--output", type=Path, help="output PNG path")
    gen_random.add_argument(
        "--no-save", action="store_true", help="do not write the recipe beside the image"
    )

    listing = sub.add_parser("list", help="list generators or palettes")
    listing.add_argument(
        "what", nargs="?", choices=("generators", "palettes"), default="generators"
    )
    listing.add_argument(
        "--params", action="store_true", help="show parameters and defaults"
    )
    listing.add_argument("--json", action="store_true")

    describe = sub.add_parser("describe", help="show how a recipe resolves")
    describe.add_argument("recipe", type=Path)
    describe.add_argument("--json", action="store_true")

    return parser


def _coerce(text: str) -> Any:
    """Read a command-line value as JSON where possible, else as a string."""
    stripped = text.strip()
    lowered = stripped.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return text


def _parse_assignments(pairs: Sequence[str], label: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"{label} override {pair!r} must look like KEY=VALUE")
        key, _, value = pair.partition("=")
        key = key.strip()
        if not key:
            raise ValueError(f"{label} override {pair!r} has an empty key")
        result[key] = _coerce(value)
    return result


def _clean(exc: Exception) -> str:
    """Registry lookups raise KeyError, whose str() adds quotes."""
    message = str(exc)
    return message[1:-1] if message.startswith("'") and message.endswith("'") else message


def _apply_overrides(recipe: Recipe, args: argparse.Namespace) -> Recipe:
    try:
        changes = _parse_assignments(args.set, "recipe")
        param_overrides = _parse_assignments(args.param, "param")
        post_overrides = _parse_assignments(args.post, "post")
    except ValueError as exc:
        raise ValueError(str(exc)) from exc

    if param_overrides:
        changes["params"] = {**recipe.params, **param_overrides}
    if post_overrides:
        changes["post"] = {**recipe.post, **post_overrides}

    updated = recipe.with_changes(**changes)
    if args.scale:
        updated = updated.scaled(args.scale)
    if args.preview:
        updated = render_module.preview_size(updated, args.preview_edge)
    return updated


def cmd_gen(args: argparse.Namespace) -> int:
    if args.recipe is None:
        print("genesis gen: a recipe path is required (or use 'gen random')", file=sys.stderr)
        return 2

    try:
        recipe = Recipe.load(args.recipe)
    except (OSError, ValueError) as exc:
        return _error(f"{args.recipe}: {_clean(exc)}")

    try:
        recipe = _apply_overrides(recipe, args)
    except (ValueError, KeyError) as exc:
        return _error(_clean(exc))

    target = args.output or args.recipe.with_name(f"{recipe.slug()}.png")
    if not args.quiet and not args.json:
        print(
            f"rendering {recipe.generator} {recipe.width}x{recipe.height} "
            f"seed={recipe.effective_seed}",
            file=sys.stderr,
        )
    try:
        result = render_module.render_to_file(recipe, target)
    except (ValueError, KeyError) as exc:
        return _error(_clean(exc))

    if args.save:
        recipe.save(args.recipe)

    if args.json:
        print(
            json.dumps(
                {
                    "image": str(target),
                    "digest": result.digest,
                    "generator": recipe.generator,
                    "seed": recipe.effective_seed,
                    "width": recipe.width,
                    "height": recipe.height,
                },
                indent=2,
            )
        )
    elif not args.quiet:
        print(f"wrote {target}  sha256={result.digest[:16]}")
    return 0


def cmd_gen_random(args: argparse.Namespace) -> int:
    entropy = stream(args.seed, "entropy") if args.seed is not None else None
    if entropy is None:
        entropy = stream(random_seed(stream(0, "entropy")), "params")

    overrides: dict[str, Any] = {
        "width": args.width,
        "height": args.height,
        "theme": "light" if args.light else "dark",
        "seed": args.seed,
    }
    if args.palette:
        overrides["palette_name"] = args.palette

    try:
        recipe = random_recipe(args.generator, rng=entropy, **overrides)
    except (ValueError, KeyError) as exc:
        return _error(_clean(exc))

    if args.preview:
        recipe = render_module.preview_size(recipe, PREVIEW_EDGE)

    target = args.output or Path(f"{recipe.slug()}.png")
    try:
        result = render_module.render_to_file(recipe, target)
    except (ValueError, KeyError) as exc:
        return _error(_clean(exc))

    recipe_path: Path | None = None
    if not args.no_save:
        recipe_path = target.with_suffix(".json")
        recipe.save(recipe_path)

    print(
        json.dumps(
            {
                "recipe": str(recipe_path) if recipe_path else None,
                "image": str(target),
                "digest": result.digest,
                "generator": recipe.generator,
                "palette": recipe.palette_name,
                "seed": recipe.effective_seed,
            },
            indent=2,
        )
    )
    return 0


def _print_param(spec: ParamSpec) -> None:
    bounds = ""
    if spec.choices:
        bounds = f"  choices: {', '.join(spec.choices)}"
    elif spec.minimum is not None or spec.maximum is not None:
        bounds = f"  range: {spec.minimum}..{spec.maximum}"
    print(f"    {spec.name:14s} {spec.kind:6s} = {spec.default!r}{bounds}")
    if spec.help:
        print(f"                   {spec.help}")


def cmd_list(args: argparse.Namespace) -> int:
    registry.ensure_loaded()

    if args.what == "palettes":
        if args.json:
            print(json.dumps(PRESETS, indent=2))
            return 0
        for name, colors in sorted(PRESETS.items()):
            print(f"{name:8s} {' '.join(colors)}")
        return 0

    if args.json:
        print(json.dumps([g.to_json() for g in registry.all_generators()], indent=2))
        return 0

    for generator in registry.all_generators():
        print(f"{generator.name}  [{generator.family}]")
        print(f"  {generator.summary}")
        print(f"  {len(generator.schema)} parameters")
        if args.params:
            for spec in generator.schema:
                _print_param(spec)
        print()
    return 0


def cmd_describe(args: argparse.Namespace) -> int:
    try:
        recipe = Recipe.load(args.recipe)
    except (OSError, ValueError) as exc:
        return _error(f"{args.recipe}: {_clean(exc)}")

    payload = render_module.describe(recipe)
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"generator  {payload['generator']} ({payload['family']})")
    print(f"size       {payload['width']}x{payload['height']}")
    print(f"seed       {payload['seed']}  (hex {payload['seed']:016x})")
    print(f"palette    {payload['palette_name']}  {' '.join(payload['palette'])}")
    print(f"theme      {payload['theme']}")
    print(f"slug       {payload['slug']}")
    print("parameters")
    for key, value in sorted(payload["params"].items()):
        print(f"  {key:14s} {value!r}")
    return 0


HANDLERS = {
    "gen": cmd_gen,
    "gen-random": cmd_gen_random,
    "list": cmd_list,
    "describe": cmd_describe,
}


def main(argv: Sequence[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    # `genesis gen random` reads better than `gen-random`, and subparsers cannot
    # have multi-word names, so rewrite the alias before parsing.
    if len(tokens) >= 2 and tokens[:2] == ["gen", "random"]:
        tokens = ["gen-random", *tokens[2:]]

    args = build_parser().parse_args(tokens)
    registry.ensure_loaded()
    try:
        return HANDLERS[args.command](args)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
