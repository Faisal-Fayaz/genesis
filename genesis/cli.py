"""Command-line interface.

Rendering verbs: `gen` renders a recipe, `gen random` invents one, `list` shows
what exists, and `describe` shows how a recipe resolves.

Walls of the app: `apply` sets the desktop wallpaper, `daily` renders or reuses
one and applies it, `gallery` serves the browser interface, and `library`
manages what has been rendered.

Overrides are layered onto a loaded recipe and the result is revalidated, so an
override that would produce an invalid render fails immediately with a useful
message rather than partway through a render.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
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
    gen_random.add_argument(
        "--post",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a post option, e.g. --post grain=0.04",
    )
    gen_random.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a generator parameter, e.g. --param trails=8000",
    )
    gen_random.add_argument(
        "--no-save", action="store_true", help="do not write the recipe beside the image"
    )
    gen_random.add_argument(
        "-q", "--quiet", action="store_true", help="print only the image path"
    )
    gen_random.add_argument("-o", "--output", type=Path, help="output PNG path")

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

    apply_cmd = sub.add_parser("apply", help="set a rendered image as the wallpaper")
    apply_cmd.add_argument(
        "target",
        nargs="?",
        help="image path, recipe digest, or recipe slug (default: library's latest)",
    )
    apply_cmd.add_argument(
        "--theme",
        choices=("dark", "light", "both"),
        default="dark",
        help="which GNOME theme's wallpaper to set (default: dark)",
    )
    apply_cmd.add_argument(
        "--recipe",
        type=Path,
        help="render this recipe first, then apply it",
    )
    apply_cmd.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a generator parameter when rendering --recipe",
    )
    apply_cmd.add_argument(
        "--post",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a post option when rendering --recipe",
    )
    apply_cmd.add_argument(
        "--no-stage",
        action="store_true",
        help="apply the file where it is, instead of copying it somewhere stable",
    )
    apply_cmd.add_argument(
        "--root", type=Path, help="override the library location"
    )

    daily = sub.add_parser("daily", help="render or reuse a wallpaper and apply it")
    daily.add_argument(
        "--mode",
        choices=("auto", "new", "reuse"),
        default="auto",
        help="auto reuses something fresh if the library has it",
    )
    daily.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    daily.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    daily.add_argument("--seed", type=int, help="pin the day's draw")
    daily.add_argument("--theme", choices=("dark", "light"), default="dark")
    daily.add_argument(
        "--dry-run",
        action="store_true",
        help="show the ranked candidates and exit without rendering",
    )
    daily.add_argument(
        "--render-only", action="store_true", help="render but do not set the wallpaper"
    )
    daily.add_argument(
        "--post",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="post option for new recipes, e.g. --post grain=0.04",
    )
    daily.add_argument(
        "--root", type=Path, help="override the library location"
    )
    daily.add_argument("--json", action="store_true")

    gallery = sub.add_parser("gallery", help="serve the local gallery")
    gallery.add_argument("--host", default="127.0.0.1")
    gallery.add_argument("--port", type=int, default=8765)
    gallery.add_argument("--root", type=Path, help="override the library location")
    gallery.add_argument(
        "--open", action="store_true", help="print the URL and open a browser"
    )

    install = sub.add_parser(
        "install", help="install and enable the daily wallpaper timer"
    )
    install.add_argument(
        "--systemd",
        action="store_true",
        help="enable the timer, which is the default when systemd is available",
    )
    install.add_argument(
        "--print", dest="print_only", action="store_true", help="print the commands"
    )

    lib = sub.add_parser("library", help="list or manage rendered pieces")
    lib.add_argument(
        "action",
        nargs="?",
        choices=("list", "star", "unstar", "delete", "prune", "show", "import"),
        default="list",
    )
    lib.add_argument("digest", nargs="?", help="digest or slug, where required")
    lib.add_argument("--limit", type=int, default=40)
    lib.add_argument("--generator", help="filter by generator")
    lib.add_argument("--palette", help="filter by palette")
    lib.add_argument("--starred", action="store_true", help="only starred pieces")
    lib.add_argument(
        "--files", action="store_true", help="delete images as well as records"
    )
    lib.add_argument("--json", action="store_true")
    lib.add_argument("--root", type=Path, help="override the library location")

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


def _with_library(args: argparse.Namespace):
    """Open a library at the requested or default location."""
    from . import library as library_module

    return library_module.Library(getattr(args, "root", None))


def _resolve_entry(library, target: str | None):
    """Find a piece by digest, then by slug, then fall back to the newest."""
    if target:
        entry = library.get(target) or library.find(target)
        if entry is None:
            raise KeyError(f"no piece matching {target!r}")
        return entry
    entry = library.latest()
    if entry is None:
        raise KeyError("the library is empty; render something first")
    return entry


def cmd_apply(args: argparse.Namespace) -> int:
    from . import library as library_module
    from . import wallpaper as wallpaper_module

    library = _with_library(args)
    library.open()
    try:
        if args.recipe:
            try:
                recipe = Recipe.load(args.recipe)
                changes: dict[str, Any] = {}
                if args.param:
                    changes["params"] = {
                        **recipe.params,
                        **_parse_assignments(args.param, "param"),
                    }
                if args.post:
                    changes["post"] = {
                        **recipe.post,
                        **_parse_assignments(args.post, "post"),
                    }
                if changes:
                    recipe = recipe.with_changes(**changes)
                result = render_module.render(recipe)
                library_module.store(library, recipe, result)
                image_path = library.find(recipe.slug()).image_path
            except (OSError, ValueError, KeyError) as exc:
                return _error(_clean(exc))
        else:
            try:
                entry = _resolve_entry(library, args.target)
            except KeyError as exc:
                return _error(_clean(exc))
            image_path = entry.image_path

        try:
            state = wallpaper_module.apply(
                image_path, theme=args.theme, stage_copy=not args.no_stage
            )
        except wallpaper_module.WallpaperUnavailable as exc:
            return _error(str(exc))
        except (FileNotFoundError, OSError) as exc:
            return _error(str(exc))
    finally:
        library.close()

    if not wallpaper_module.fits_display(image_path_size(image_path)):
        print(
            f"note: {image_path.name} is smaller than the display and will be scaled",
            file=sys.stderr,
        )
    print(f"wallpaper set from {image_path.name}")
    print(f"  dark:  {state.dark}")
    print(f"  light: {state.light}")
    return 0


def image_path_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        return image.size


def cmd_daily(args: argparse.Namespace) -> int:
    from . import library as library_module
    from . import scheduler as scheduler_module
    from . import wallpaper as wallpaper_module

    try:
        post = _parse_assignments(args.post, "post")
    except ValueError as exc:
        return _error(str(exc))

    library = _with_library(args)
    library.open()
    try:
        if args.dry_run:
            ranked = scheduler_module.rank(library.list(limit=1000))
            if args.json:
                print(
                    json.dumps(
                        [
                            {
                                "digest": choice.entry.digest,
                                "slug": choice.entry.slug,
                                "score": round(choice.score, 2),
                                "reasons": {k: round(v, 2) for k, v in choice.reasons.items()},
                            }
                            for choice in ranked
                        ],
                        indent=2,
                    )
                )
                return 0
            if not ranked:
                print("the library is empty; the next run will generate one")
                return 0
            print(f"{len(ranked)} candidates:")
            print(scheduler_module.summary(ranked))
            return 0

        try:
            plan = scheduler_module.choose(
                library,
                width=args.width,
                height=args.height,
                mode=args.mode,
                seed=args.seed,
                theme=args.theme,
                post=post,
            )
        except ValueError as exc:
            return _error(_clean(exc))

        if plan.existing is not None:
            entry = plan.existing.entry
            if args.render_only:
                print(f"would reuse {entry.slug} ({plan.reason})")
                return 0
            try:
                wallpaper_module.apply(entry.image_path, theme="both")
            except wallpaper_module.WallpaperUnavailable as exc:
                print(f"skipped wallpaper change: {exc}", file=sys.stderr)
                print(f"would reuse {entry.slug}", file=sys.stderr)
                return 0
            library.record_applied(entry.digest)
            print(f"reused {entry.slug}  ({plan.reason})")
            return 0

        started = time.perf_counter()
        try:
            result = render_module.render(plan.recipe)
        except (ValueError, KeyError) as exc:
            return _error(_clean(exc))
        elapsed = time.perf_counter() - started

        entry = library_module.store(library, plan.recipe, result, elapsed)
        print(f"rendered {entry.slug} in {elapsed:.1f}s  ({plan.reason})")

        if args.render_only:
            print(f"  saved to {entry.image_path}")
            return 0

        try:
            wallpaper_module.apply(entry.image_path, theme="both")
        except wallpaper_module.WallpaperUnavailable as exc:
            print(f"skipped wallpaper change: {exc}", file=sys.stderr)
            print(f"  saved to {entry.image_path}", file=sys.stderr)
            return 0
        library.record_applied(entry.digest)
        print("wallpaper set")
        return 0
    finally:
        library.close()


def cmd_gallery(args: argparse.Namespace) -> int:
    from . import server as server_module

    try:
        server = server_module.build_server(args.host, args.port, args.root)
    except ValueError as exc:
        return _error(str(exc))
    except OSError as exc:
        return _error(f"cannot bind {args.host}:{args.port}: {exc}")

    url = f"http://{args.host}:{server.server_address[1]}/"
    print(f"gallery on {url}")
    print(f"library {server.library.root}")
    print("ctrl-c to stop")

    if args.open:
        import webbrowser

        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()
        server.library.close()
    return 0


def systemd_available() -> bool:
    """Whether a systemd user manager can be reached."""
    if not shutil.which("systemctl"):
        return False
    completed = subprocess.run(
        ["systemctl", "--user", "is-system-running"],
        capture_output=True,
        text=True,
        check=False,
    )
    # `degraded` and `running` both mean the manager is up.
    return completed.returncode == 0 or "running" in completed.stdout


def unit_names() -> list[str]:
    from . import systemd as systemd_module

    return systemd_module.UNIT_NAMES


def cmd_install(args: argparse.Namespace) -> int:
    """Copy the timer units into place and enable them."""
    from . import systemd as systemd_module

    if args.print_only:
        for line in systemd_module.install_commands():
            print(line)
        return 0

    if not systemd_available():
        return _error(
            "no systemd user manager found; run 'genesis install --print' "
            "and run the commands by hand"
        )

    try:
        copied = systemd_module.install()
    except OSError as exc:
        return _error(str(exc))

    for name in copied:
        print(f"installed {name}")

    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    subprocess.run(
        ["systemctl", "--user", "enable", "--now", f"{systemd_module.TIMER_STEM}.timer"],
        check=False,
    )
    print()
    print("the timer is enabled; check it with:")
    print(f"  systemctl --user list-timers {systemd_module.TIMER_STEM}.timer")
    print()
    print("run today's wallpaper now with:")
    print("  genesis daily --dry-run    # see the ranked candidates")
    print("  genesis daily              # render and apply")
    return 0


def cmd_library(args: argparse.Namespace) -> int:

    library = _with_library(args)
    library.open()
    try:
        return _run_library_action(library, args)
    finally:
        library.close()


def _run_library_action(library, args: argparse.Namespace) -> int:
    from . import library as library_module

    if args.action == "list":
        entries = library.list(
            limit=args.limit,
            generator=args.generator,
            palette=args.palette,
            starred=True if args.starred else None,
        )
        if args.json:
            print(json.dumps([entry.to_json() for entry in entries], indent=2))
            return 0
        if not entries:
            print("the library is empty")
            return 0
        print(f"{len(entries)} of {library.count()} pieces:")
        for entry in entries:
            print("  " + library_module.describe_entry(entry))
        return 0

    if args.action in ("star", "unstar"):
        try:
            entry = _resolve_entry(library, args.digest)
        except KeyError as exc:
            return _error(_clean(exc))
        library.set_starred(entry.digest, args.action == "star")
        print(f"{'starred' if args.action == 'star' else 'unstarred'} {entry.slug}")
        return 0

    if args.action == "delete":
        try:
            entry = _resolve_entry(library, args.digest)
        except KeyError as exc:
            return _error(_clean(exc))
        library.delete(entry.digest, remove_files=args.files)
        print(f"deleted {entry.slug}" + (" and its images" if args.files else ""))
        return 0

    if args.action == "prune":
        removed = library.prune_orphans(remove_files=args.files)
        print(f"removed {removed} record(s) whose images were missing")
        return 0

    if args.action == "show":
        try:
            entry = _resolve_entry(library, args.digest)
        except KeyError as exc:
            return _error(_clean(exc))
        if args.json:
            print(json.dumps(entry.to_json(), indent=2))
            return 0
        print(f"digest    {entry.digest}")
        print(f"slug      {entry.slug}")
        print(f"generator {entry.generator}")
        print(f"size      {entry.width}x{entry.height}")
        print(f"palette   {entry.palette_name}")
        print(f"theme     {entry.theme}")
        print(f"rendered  {library_module.elapsed(entry.render_seconds)}")
        print(f"starred   {'yes' if entry.starred else 'no'}")
        print(f"applied   {entry.applied_count} time(s)")
        print(f"image     {entry.image_path}")
        if entry.preview_path:
            print(f"preview   {entry.preview_path}")
        return 0

    if args.action == "import":
        """Register images rendered outside the CLI, by re-deriving digests."""
        source = Path(args.digest or ".")
        added = 0
        for recipe_path in sorted(source.glob("**/*.json")):
            try:
                recipe = Recipe.load(recipe_path)
                result = render_module.render(recipe)
            except (OSError, ValueError, KeyError):
                continue
            library_module.store(library, recipe, result)
            added += 1
        print(f"imported {added} recipe(s) from {source}")
        return 0

    return _error(f"unhandled library action {args.action!r}")


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
        changes: dict[str, Any] = {}
        if args.param:
            changes["params"] = {**recipe.params, **_parse_assignments(args.param, "param")}
        if args.post:
            changes["post"] = {**recipe.post, **_parse_assignments(args.post, "post")}
        if changes:
            recipe = recipe.with_changes(**changes)
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

    if args.quiet:
        print(target)
        return 0

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
    "apply": cmd_apply,
    "install": cmd_install,
    "daily": cmd_daily,
    "gallery": cmd_gallery,
    "library": cmd_library,
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
