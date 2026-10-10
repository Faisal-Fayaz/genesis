"""Installing the timer.

The units themselves are checked in `test_systemd_units.py`. This covers the
copy-and-enable path, including the refusal when there is no user manager, which
is the case on a machine without systemd or over a plain ssh session.
"""

from __future__ import annotations

import subprocess

import pytest

from genesis import cli
from genesis import systemd as systemd_module


class TestUnitDiscovery:
    def test_both_units_are_present(self) -> None:
        for name in systemd_module.UNIT_NAMES:
            assert systemd_module.unit_path(name).exists(), name

    def test_unit_names_share_a_stem(self) -> None:
        assert systemd_module.SERVICE_NAME == "genesis-daily.service"
        assert systemd_module.TIMER_NAME == "genesis-daily.timer"

    def test_install_commands_cover_every_unit(self) -> None:
        commands = " ".join(systemd_module.install_commands())
        for name in systemd_module.UNIT_NAMES:
            assert name in commands

    def test_install_commands_reload_and_enable(self) -> None:
        commands = " ".join(systemd_module.install_commands())
        assert "daemon-reload" in commands
        assert "enable --now" in commands


class TestInstall:
    def test_copies_both_units(self, tmp_path) -> None:
        copied = systemd_module.install(tmp_path)
        assert sorted(copied) == sorted(systemd_module.UNIT_NAMES)
        for name in systemd_module.UNIT_NAMES:
            assert (tmp_path / name).exists()

    def test_copied_contents_match_the_rendered_units(self, tmp_path) -> None:
        """Installed units point at the resolved executable, not the checkout.

        Only the ExecStart line may differ from the shipped file; everything
        else must be byte-identical so installer edits stay reviewable.
        """
        systemd_module.install(tmp_path)
        executable = systemd_module.resolve_executable()
        for name in systemd_module.UNIT_NAMES:
            installed = (tmp_path / name).read_text()
            if name.endswith(".service"):
                assert installed == systemd_module.render_unit(name, executable)
                assert f"ExecStart={executable}" in installed
            else:
                # The timer schedules; it invokes nothing, so it copies verbatim.
                assert installed == systemd_module.unit_path(name).read_text()

    def test_creates_the_destination(self, tmp_path) -> None:
        target = tmp_path / "nested" / "user"
        systemd_module.install(target)
        assert target.is_dir()

    def test_reports_what_is_installed(self, tmp_path) -> None:
        assert systemd_module.is_installed(tmp_path) is False
        systemd_module.install(tmp_path)
        assert systemd_module.is_installed(tmp_path) is True

    def test_overwrites_an_existing_unit(self, tmp_path) -> None:
        (tmp_path / systemd_module.SERVICE_NAME).write_text("stale")
        systemd_module.install(tmp_path)
        assert "genesis" in (tmp_path / systemd_module.SERVICE_NAME).read_text()

    def test_refuses_without_systemctl(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(systemd_module.shutil, "which", lambda name: None)
        with pytest.raises(OSError, match="systemctl"):
            systemd_module.install(tmp_path)


class TestTimerStatus:
    def test_reports_the_enabled_state(self, monkeypatch) -> None:
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="enabled\n", stderr=""
        )
        monkeypatch.setattr(systemd_module.subprocess, "run", lambda *a, **k: completed)
        enabled, status = systemd_module.timer_status()
        assert enabled is True
        assert status == "enabled"

    def test_reports_a_disabled_timer(self, monkeypatch) -> None:
        completed = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="disabled\n", stderr=""
        )
        monkeypatch.setattr(systemd_module.subprocess, "run", lambda *a, **k: completed)
        enabled, _status = systemd_module.timer_status()
        assert enabled is False

    def test_next_run_is_read_from_systemd(self, monkeypatch) -> None:
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="NextElapseUSecRealtime=Thu 2026\n", stderr=""
        )
        monkeypatch.setattr(systemd_module.subprocess, "run", lambda *a, **k: completed)
        assert systemd_module.next_run() == "Thu 2026"

    def test_next_run_falls_back_to_unknown(self, monkeypatch) -> None:
        completed = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="no such unit"
        )
        monkeypatch.setattr(systemd_module.subprocess, "run", lambda *a, **k: completed)
        assert systemd_module.next_run() == "unknown"


class TestInstallCommand:
    def test_print_lists_the_commands(self, capsys) -> None:
        code = cli.main(["install", "--print"])
        out = capsys.readouterr().out
        assert code == 0
        assert "systemctl --user enable" in out

    def test_refuses_without_a_user_manager(self, capsys, monkeypatch) -> None:
        monkeypatch.setattr(cli, "systemd_available", lambda: False)
        code = cli.main(["install"])
        err = capsys.readouterr().err
        assert code == 1
        assert "no systemd user manager" in err

    def test_copies_and_enables(self, capsys, monkeypatch, tmp_path) -> None:
        calls: list[list[str]] = []
        monkeypatch.setattr(cli, "systemd_available", lambda: True)
        monkeypatch.setattr(
            systemd_module, "install", lambda destination=None, live=False, app=False: list(systemd_module.UNIT_NAMES)
        )
        monkeypatch.setattr(
            systemd_module.subprocess,
            "run",
            lambda args, **kwargs: calls.append(args) or subprocess.CompletedProcess(
                args, 0, "", ""
            ),
        )
        code = cli.main(["install"])
        out = capsys.readouterr().out
        assert code == 0
        assert "installed" in out
        assert any("daemon-reload" in " ".join(call) for call in calls)
        assert any("enable" in " ".join(call) for call in calls)

    def test_reports_a_copy_failure(self, capsys, monkeypatch) -> None:
        monkeypatch.setattr(cli, "systemd_available", lambda: True)

        def fail(destination=None, live=False, app=False):
            raise OSError("read-only filesystem")

        monkeypatch.setattr(systemd_module, "install", fail)
        code = cli.main(["install"])
        err = capsys.readouterr().err
        assert code == 1
        assert "read-only filesystem" in err

    def test_points_at_the_cli_in_its_own_messages(self, capsys, monkeypatch) -> None:
        monkeypatch.setattr(cli, "systemd_available", lambda: True)
        monkeypatch.setattr(
            systemd_module, "install", lambda destination=None, live=False, app=False: list(systemd_module.UNIT_NAMES)
        )
        monkeypatch.setattr(
            systemd_module.subprocess,
            "run",
            lambda args, **kwargs: subprocess.CompletedProcess(args, 0, "", ""),
        )
        cli.main(["install"])
        out = capsys.readouterr().out
        assert "genesis daily" in out


class TestServiceExecStart:
    def test_uses_the_project_interpreter(self) -> None:
        """A bare `python` would resolve to the system one, which lacks numpy."""
        text = systemd_module.unit_path(systemd_module.SERVICE_NAME).read_text()
        assert "%h/projects/genesis/.venv/bin/python" in text

    def test_runs_the_daily_command(self) -> None:
        text = systemd_module.unit_path(systemd_module.SERVICE_NAME).read_text()
        assert "-m genesis.cli daily" in text


class TestNoDestructiveDefaults:
    def test_installing_units_does_not_enable_them(self) -> None:
        """Installing files must not change the desktop on its own.

        `install` explicitly enables the timer, which is the user's decision to
        run; nothing else in the package may enable it.
        """
        unit_text = systemd_module.unit_path(systemd_module.TIMER_NAME).read_text()
        assert "WantedBy=timers.target" in unit_text
        # The timer's schedule is the only trigger; no unit may start it directly.
        service_text = systemd_module.unit_path(systemd_module.SERVICE_NAME).read_text()
        assert "genesis-daily.timer" not in service_text


class TestExecutableResolution:
    def test_prefers_genesis_on_path(self, monkeypatch, tmp_path) -> None:
        exe = tmp_path / "bin" / "genesis"
        exe.parent.mkdir()
        exe.write_text("#!/bin/sh\n")
        monkeypatch.setattr(systemd_module.shutil, "which", lambda name: str(exe))
        assert systemd_module.resolve_executable() == str(exe)

    def test_falls_back_to_the_current_interpreter(self, monkeypatch) -> None:
        monkeypatch.setattr(systemd_module.shutil, "which", lambda name: None)
        resolved = systemd_module.resolve_executable()
        assert resolved.endswith(" -m genesis.cli")
        assert "genesis.cli" in resolved

    def test_render_unit_rewrites_only_exec_start(self) -> None:
        source = systemd_module.unit_path(systemd_module.SERVICE_NAME).read_text()
        rendered = systemd_module.render_unit(
            systemd_module.SERVICE_NAME, "/home/u/.local/bin/genesis"
        )
        assert "ExecStart=/home/u/.local/bin/genesis daily" in rendered
        source_lines = [line for line in source.splitlines() if not line.startswith("ExecStart=")]
        rendered_lines = [line for line in rendered.splitlines() if not line.startswith("ExecStart=")]
        assert source_lines == rendered_lines

    def test_render_unit_rejects_units_without_a_genesis_command(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="no ExecStart"):
            systemd_module.render_unit("genesis-daily.timer", "/bin/genesis")


class TestAppInstallation:
    def test_install_app_writes_desktop_entry_and_icon(self, tmp_path) -> None:
        apps = tmp_path / "applications"
        icons = tmp_path / "icons"
        names = systemd_module.install_app_files(apps, icons, "/bin/genesis")
        assert names == ["genesis.desktop", "genesis.svg"]
        entry = (apps / "genesis.desktop").read_text()
        assert "Exec=/bin/genesis app" in entry
        assert "@GENESIS_EXEC@" not in entry
        assert "StartupWMClass=org.genesis.wallpapers" in entry
        assert (icons / "genesis.svg").exists()

    def test_install_app_flag_adds_desktop_files(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(
            systemd_module, "install_app_files",
            lambda *a, **k: ["genesis.desktop", "genesis.svg"],
        )
        copied = systemd_module.install(tmp_path, app=True)
        assert "genesis.desktop" in copied

    def test_install_without_app_flag_skips_desktop_files(self, tmp_path, monkeypatch) -> None:
        calls: list = []
        monkeypatch.setattr(
            systemd_module, "install_app_files",
            lambda *a, **k: calls.append(True) or [],
        )
        systemd_module.install(tmp_path)
        assert calls == []

    def test_install_print_lists_app_steps(self, capsys) -> None:
        code = cli.main(["install", "--print", "--app"])
        assert code == 0
        out = capsys.readouterr().out
        assert "genesis.desktop" in out
        assert "genesis.svg" in out