"""The gallery server.

Requests go over real HTTP against a real server on an ephemeral port, so
routing, status codes, headers and content types are all covered rather than just
the handler functions. The binding restriction gets its own tests because it is
a deliberate refusal, not an accident.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

from genesis import library as library_module
from genesis import render as render_module
from genesis import server as server_module
from genesis.recipe import Recipe


@pytest.fixture
def gallery(tmp_path) -> Iterator[str]:
    server, _thread = server_module.serve_in_thread(port=0, root=tmp_path / "lib")
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        server.library.close()


def get(base: str, path: str) -> tuple[int, str, bytes]:
    with urllib.request.urlopen(base + path) as response:
        return response.status, response.headers.get("Content-Type", ""), response.read()


def post(base: str, path: str, body: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        return response.status, json.loads(response.read())


def expect_error(base: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    try:
        if body is None:
            urllib.request.urlopen(base + path)
        else:
            request = urllib.request.Request(
                base + path,
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(request)
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())
    raise AssertionError(f"{path} unexpectedly succeeded")


def small_recipe(**overrides) -> Recipe:
    """A small, fast recipe. Generator-specific params follow the choice."""
    generator = overrides.get("generator", "flowfield")
    if generator == "attractor":
        params: dict[str, object] = {"points": 20000, "warmup": 50, "supersample": 1}
    else:
        params = {"trails": 1000, "steps": 25, "supersample": 1}
    fields: dict[str, object] = {
        "generator": generator,
        "params": params,
        "width": 160,
        "height": 90,
        "seed": 5,
        "palette_name": "tide",
    }
    fields.update(overrides)
    return Recipe(**fields)


def render_once(base: str, **overrides) -> dict:
    body = small_recipe(**overrides).to_dict() | {"preview": True}
    _status, payload = post(base, "/api/render", body)
    return payload["piece"]


class TestBinding:
    def test_refuses_a_non_loopback_host(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="localhost"):
            server_module.build_server("0.0.0.0", 0, tmp_path)

    def test_accepts_loopback(self, tmp_path) -> None:
        server = server_module.build_server("127.0.0.1", 0, tmp_path)
        try:
            assert server.server_address[0] == "127.0.0.1"
        finally:
            server.server_close()
            server.library.close()


class TestStaticFiles:
    def test_serves_the_page(self, gallery) -> None:
        status, content_type, body = get(gallery, "/")
        assert status == 200
        assert "text/html" in content_type
        assert b"Genesis" in body

    def test_serves_the_script(self, gallery) -> None:
        status, content_type, body = get(gallery, "/app.js")
        assert status == 200
        assert "javascript" in content_type
        assert b"api/render" in body

    def test_serves_the_stylesheet(self, gallery) -> None:
        status, content_type, _body = get(gallery, "/style.css")
        assert status == 200
        assert "text/css" in content_type

    def test_assets_are_packaged_with_the_module(self) -> None:
        for name in ("index.html", "app.js", "style.css"):
            assert (server_module.WEB_ROOT / name).exists(), name


class TestMetadata:
    def test_generators_include_parameter_metadata(self, gallery) -> None:
        _status, _ct, body = get(gallery, "/api/generators")
        payload = json.loads(body)
        names = {item["name"] for item in payload}
        assert names >= {"flowfield", "attractor"}
        flowfield = next(i for i in payload if i["name"] == "flowfield")
        assert any(p["kind"] == "int" for p in flowfield["params"])
        assert any("minimum" in p for p in flowfield["params"])

    def test_palettes_are_listed(self, gallery) -> None:
        _status, _ct, body = get(gallery, "/api/palettes")
        assert "tide" in json.loads(body)

    def test_post_schema_matches_the_declarations(self, gallery) -> None:
        from genesis import post as post_module

        _status, _ct, body = get(gallery, "/api/post-schema")
        payload = json.loads(body)
        assert [p["name"] for p in payload["post"]] == [
            s.name for s in post_module.SCHEMA
        ]

    def test_wallpaper_state_is_reported(self, gallery) -> None:
        _status, _ct, body = get(gallery, "/api/wallpaper")
        payload = json.loads(body)
        assert "session" in payload
        assert "dark" in payload


class TestPieces:
    def test_empty_library_lists_nothing(self, gallery) -> None:
        _status, _ct, body = get(gallery, "/api/pieces")
        payload = json.loads(body)
        assert payload["pieces"] == []
        assert payload["count"] == 0

    def test_rendered_piece_appears(self, gallery) -> None:
        render_once(gallery)
        _status, _ct, body = get(gallery, "/api/pieces")
        payload = json.loads(body)
        assert payload["count"] == 1
        assert payload["pieces"][0]["generator"] == "flowfield"

    def test_piece_carries_urls(self, gallery) -> None:
        piece = render_once(gallery)
        _status, _ct, body = get(gallery, "/api/pieces")
        listed = json.loads(body)["pieces"][0]
        assert listed["image"] == f"/image/{piece['digest']}"
        assert listed["recipe"]["width"] == 160

    def test_generator_filter(self, gallery) -> None:
        render_once(gallery)
        render_once(gallery, seed=6)
        render_once(gallery, generator="attractor")
        _status, _ct, body = get(gallery, "/api/pieces?generator=attractor")
        payload = json.loads(body)
        assert payload["count"] == 1
        assert payload["pieces"][0]["generator"] == "attractor"

    def test_filter_reports_matched_and_total(self, gallery) -> None:
        render_once(gallery)
        render_once(gallery, seed=6, palette_name="ember")
        _status, _ct, body = get(gallery, "/api/pieces?palette=ember")
        payload = json.loads(body)
        assert payload["count"] == 1
        assert payload["total"] == 2

    def test_palette_filter(self, gallery) -> None:
        render_once(gallery)
        render_once(gallery, seed=6, palette_name="ember")
        _status, _ct, body = get(gallery, "/api/pieces?palette=ember")
        assert json.loads(body)["count"] == 1

    def test_limit_is_honoured(self, gallery) -> None:
        for seed in (1, 2, 3):
            render_once(gallery, seed=seed)
        _status, _ct, body = get(gallery, "/api/pieces?limit=2")
        assert len(json.loads(body)["pieces"]) == 2


class TestRender:
    def test_renders_and_records(self, gallery) -> None:
        status, payload = post(gallery, "/api/render", small_recipe().to_dict())
        assert status == 200
        assert len(payload["piece"]["digest"]) == 64
        assert payload["render_seconds"] >= 0

    def test_preview_flag_is_not_treated_as_a_recipe_field(self, gallery) -> None:
        body = small_recipe(width=1920, height=1080).to_dict() | {"preview": True}
        status, payload = post(gallery, "/api/render", body)
        assert status == 200
        assert payload["piece"]["width"] == 640

    def test_preview_is_smaller_than_the_request(self, gallery) -> None:
        body = small_recipe(width=1920, height=1080).to_dict() | {"preview": True}
        _status, payload = post(gallery, "/api/render", body)
        assert max(payload["piece"]["width"], payload["piece"]["height"]) == 640

    def test_a_large_render_gets_a_preview_image(self, gallery) -> None:
        body = small_recipe(width=1920, height=1080).to_dict() | {"preview": True}
        _status, payload = post(gallery, "/api/render", body)
        assert payload["piece"]["preview"] is not None

    def test_rendering_the_same_recipe_twice_does_not_duplicate(self, gallery) -> None:
        recipe = small_recipe().to_dict()
        post(gallery, "/api/render", recipe)
        post(gallery, "/api/render", recipe)
        _status, _ct, body = get(gallery, "/api/pieces")
        assert json.loads(body)["count"] == 1

    def test_invalid_parameters_are_a_bad_request(self, gallery) -> None:
        recipe = small_recipe().to_dict()
        recipe["params"]["trails"] = 0
        status, payload = expect_error(gallery, "/api/render", recipe)
        assert status == 400
        assert "minimum" in payload["error"]

    def test_unknown_generator_is_a_bad_request(self, gallery) -> None:
        recipe = small_recipe().to_dict() | {"generator": "nonexistent"}
        status, payload = expect_error(gallery, "/api/render", recipe)
        assert status == 400
        assert "available" in payload["error"]

    def test_unknown_field_is_rejected(self, gallery) -> None:
        status, payload = expect_error(
            gallery, "/api/render", small_recipe().to_dict() | {"bogus": 1}
        )
        assert status == 400
        assert "unknown recipe field" in payload["error"]


class TestImages:
    def test_serves_the_full_image(self, gallery) -> None:
        piece = render_once(gallery)
        status, content_type, body = get(gallery, f"/image/{piece['digest']}")
        assert status == 200
        assert content_type == "image/png"
        assert body[:8] == b"\x89PNG\r\n\x1a\n"

    def test_serves_a_preview(self, gallery) -> None:
        body = small_recipe(width=1920, height=1080).to_dict() | {"preview": True}
        _status, payload = post(gallery, "/api/render", body)
        digest = payload["piece"]["digest"]
        status, content_type, data = get(gallery, f"/preview/{digest}")
        assert status == 200
        assert content_type == "image/png"
        assert data[:8] == b"\x89PNG\r\n\x1a\n"

    def test_preview_falls_back_to_the_full_image(self, gallery) -> None:
        piece = render_once(gallery)
        status, _ct, _body = get(gallery, f"/preview/{piece['digest']}")
        assert status == 200

    def test_unknown_digest_is_not_found(self, gallery) -> None:
        status, payload = expect_error(gallery, "/image/deadbeef")
        assert status == 404
        assert "unknown digest" in payload["error"]


class TestStarring:
    def test_toggles(self, gallery) -> None:
        piece = render_once(gallery)
        _status, first = post(gallery, "/api/star", {"digest": piece["digest"]})
        assert first["starred"] is True
        _status, second = post(gallery, "/api/star", {"digest": piece["digest"]})
        assert second["starred"] is False

    def test_star_survives_a_relist(self, gallery) -> None:
        piece = render_once(gallery)
        post(gallery, "/api/star", {"digest": piece["digest"]})
        _status, _ct, body = get(gallery, "/api/pieces")
        assert json.loads(body)["pieces"][0]["starred"] is True

    def test_starred_filter(self, gallery) -> None:
        plain = render_once(gallery)
        starred = render_once(gallery, seed=6)
        post(gallery, "/api/star", {"digest": starred["digest"]})
        _status, _ct, body = get(gallery, "/api/pieces?starred=1")
        listed = json.loads(body)["pieces"]
        assert [p["digest"] for p in listed] == [starred["digest"]]
        assert plain["digest"] != starred["digest"]

    def test_missing_digest_is_a_bad_request(self, gallery) -> None:
        status, payload = expect_error(gallery, "/api/star", {})
        assert status == 400
        assert "digest is required" in payload["error"]

    def test_unknown_digest_is_not_found(self, gallery) -> None:
        status, _payload = expect_error(gallery, "/api/star", {"digest": "nope"})
        assert status == 404


class TestApply:
    def test_reports_conflict_without_a_session(self, gallery, monkeypatch) -> None:
        from genesis import wallpaper as wallpaper_module

        piece = render_once(gallery)

        def refuse(*args, **kwargs):
            raise wallpaper_module.WallpaperUnavailable("no graphical session available")

        monkeypatch.setattr(wallpaper_module, "apply", refuse)
        status, payload = expect_error(gallery, "/api/apply", {"digest": piece["digest"]})
        assert status == 409
        assert "no graphical session" in payload["error"]

    def test_unknown_digest_is_not_found(self, gallery) -> None:
        status, _payload = expect_error(gallery, "/api/apply", {"digest": "nope"})
        assert status == 404

    def test_missing_digest_is_a_bad_request(self, gallery) -> None:
        status, _payload = expect_error(gallery, "/api/apply", {})
        assert status == 400


class TestErrors:
    def test_unknown_route_is_not_found(self, gallery) -> None:
        status, payload = expect_error(gallery, "/api/nonsense")
        assert status == 404
        assert "no such route" in payload["error"]

    def test_unknown_page_is_not_found(self, gallery) -> None:
        status, _payload = expect_error(gallery, "/nope")
        assert status == 404

    def test_malformed_json_is_a_bad_request(self, gallery) -> None:
        request = urllib.request.Request(
            gallery + "/api/render",
            data=b"{not json",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(request)
        except urllib.error.HTTPError as error:
            assert error.code == 400
            assert "not valid JSON" in json.loads(error.read())["error"]
        else:
            pytest.fail("expected a 400")


class TestConcurrency:
    def test_library_is_usable_across_request_threads(self, gallery) -> None:
        """The server is threaded, so several requests must not collide."""
        import concurrent.futures

        def fetch(index: int) -> int:
            return get(gallery, "/api/pieces")[0]

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            statuses = list(pool.map(fetch, range(16)))
        assert all(status == 200 for status in statuses)


class TestRenderHelper:
    def test_encodes_png(self) -> None:
        import numpy as np

        data = server_module.to_png_bytes(np.full((8, 8, 3), 0.5))
        assert data[:8] == b"\x89PNG\r\n\x1a\n"


class TestPackaging:
    def test_web_assets_are_declared_as_package_data(self) -> None:
        import tomllib

        config = tomllib.loads(
            (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text()
        )
        assert config["tool"]["setuptools"]["package-data"]["genesis"] == ["web/*"]


class TestLibraryIntegration:
    def test_server_and_library_share_a_root(self, tmp_path) -> None:
        root = tmp_path / "shared"
        server, _thread = server_module.serve_in_thread(port=0, root=root)
        try:
            base = f"http://127.0.0.1:{server.server_address[1]}"
            post(base, "/api/render", small_recipe().to_dict() | {"preview": True})
            assert (root / "images").exists()
            with library_module.library_at(root) as library:
                assert library.count() == 1
        finally:
            server.shutdown()
            server.server_close()
            server.library.close()

    def test_rendered_recipe_can_be_re_rendered_standalone(self, tmp_path) -> None:
        """The stored recipe must be enough to reproduce the image."""
        recipe = small_recipe()
        with library_module.library_at(tmp_path / "standalone") as library:
            library_module.store(library, recipe, render_module.render(recipe))
            stored = library.find(recipe.slug())
            reloaded = Recipe.load(stored.image_path.with_suffix(".json"))
        assert render_module.render(reloaded).digest == render_module.render(recipe).digest