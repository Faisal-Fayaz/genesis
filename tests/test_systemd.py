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

    def test_copied_contents_match(self, tmp_path) -> None:
        systemd_module.install(tmp_path)
        for name in systemd_module.UNIT_NAMES:
            assert (tmp_path / name).read_text() == (
                systemd_module.unit_path(name)
            ).read_text()

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
            systemd_module, "install", lambda destination=None: list(systemd_module.UNIT_NAMES)
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

        def fail(destination=None):
            raise OSError("read-only filesystem")

        monkeypatch.setattr(systemd_module, "install", fail)
        code = cli.main(["install"])
        err = capsys.readouterr().err
        assert code == 1
        assert "read-only filesystem" in err

    def test_points_at_the_cli_in_its_own_messages(self, capsys, monkeypatch) -> None:
        monkeypatch.setattr(cli, "systemd_available", lambda: True)
        monkeypatch.setattr(
            systemd_module, "install", lambda destination=None: list(systemd_module.UNIT_NAMES)
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