"""Doctor reports and clean uninstalls.

`genesis doctor` must run anywhere — headless CI included — and only fail
its exit code for things that make rendering impossible. `uninstall` must
remove what `install` put in place and never touch the library, and running
it twice must be a no-op rather than an error.
"""

from __future__ import annotations

import json
import shutil

from genesis import cli as cli_module
from genesis import systemd as systemd_module


def run(capsys, *argv: str) -> tuple[int, str, str]:
    code = cli_module.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


EXPECTED_CHECKS = {
    "python", "genesis", "numpy", "pillow", "library", "session",
    "display", "systemd", "app files", "gtk", "x11",
}


def test_doctor_reports_every_check(capsys) -> None:
    code, out, _ = run(capsys, "doctor")
    assert code == 0  # this environment renders, so nothing may fail
    assert "genesis doctor" in out
    report = cli_module.collect_doctor_status()
    assert {entry["check"] for entry in report} == EXPECTED_CHECKS
    assert {entry["level"] for entry in report} <= {"ok", "warn", "fail", "info"}
    by_name = {entry["check"]: entry for entry in report}
    assert by_name["python"]["level"] == "ok"


def test_doctor_json_is_parseable(capsys) -> None:
    code, out, _ = run(capsys, "doctor", "--json")
    assert code == 0
    report = json.loads(out)
    assert isinstance(report, list) and report
    assert all(set(entry) == {"check", "level", "detail"} for entry in report)


def test_uninstall_print_lists_commands(capsys) -> None:
    code, out, _ = run(capsys, "uninstall", "--print")
    assert code == 0
    assert "disable --now genesis-daily.timer" in out
    assert "genesis.desktop" in out


def test_uninstall_removes_what_install_put_down(tmp_path, monkeypatch) -> None:
    if shutil.which("systemctl") is None:
        import pytest

        pytest.skip("systemctl is not available")
    destination = tmp_path / "systemd"
    apps = tmp_path / "applications"
    icons = tmp_path / "icons"
    apps.mkdir()
    icons.mkdir()
    systemd_module.install(destination, live=True)
    (apps / systemd_module.DESKTOP_NAME).write_text("launcher")
    (icons / systemd_module.ICON_NAME).write_text("icon")

    # Never touch the real session bus from tests: file removal only.
    monkeypatch.setattr(systemd_module.shutil, "which", lambda name: None)
    removed = systemd_module.uninstall(destination, apps, icons)
    assert systemd_module.SERVICE_NAME in removed
    assert systemd_module.TIMER_NAME in removed
    assert systemd_module.LIVE_SERVICE_NAME in removed
    assert systemd_module.DESKTOP_NAME in removed
    assert systemd_module.ICON_NAME in removed
    assert not any(destination.iterdir())

    assert systemd_module.uninstall(destination, apps, icons) == []


def test_uninstall_leaves_the_library_alone(tmp_path, capsys, monkeypatch) -> None:
    from genesis import library as library_module
    from genesis import render as render_module
    from genesis.recipe import Recipe

    root = tmp_path / "lib"
    recipe = Recipe(
        generator="flowfield",
        params={"trails": 1000, "steps": 25, "supersample": 1},
        width=96,
        height=54,
        seed=7,
    )
    with library_module.library_at(root) as library:
        library_module.store(library, recipe, render_module.render(recipe))
    # Redirect every uninstall path into tmp: the real test is that the
    # library is never among the removal targets, not session-bus traffic.
    monkeypatch.setattr(systemd_module, "SYSTEMD_USER_DIR", tmp_path / "units")
    monkeypatch.setattr(systemd_module, "APPLICATIONS_DIR", tmp_path / "apps")
    monkeypatch.setattr(systemd_module, "ICONS_DIR", tmp_path / "icons")
    monkeypatch.setattr(systemd_module.shutil, "which", lambda name: None)
    code, _out, _ = run(capsys, "uninstall")
    assert code == 0
    assert (root / "library.db").exists()
    assert list((root / "images").glob("*.png"))
