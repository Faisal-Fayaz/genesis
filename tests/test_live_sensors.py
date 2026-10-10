"""Outside-world sensors: lock and battery probes fail open and stay testable.

Every outcome is scripted through an injected runner: locked, unlocked, a
missing tool, and garbage output. The contract under test is small — a probe
returns True, False, or None (unknown), and unknown must never pause a
healthy session.
"""

from __future__ import annotations

from genesis.live import sensors as sensors_module


def runner_for(code: int, out: str):
    def run(argv):
        assert argv[0] == "gdbus"
        return code, out

    return run


def test_lock_reports_locked() -> None:
    monitor = sensors_module.LockMonitor(run=runner_for(0, "(true,)"))
    assert monitor.locked() is True


def test_lock_reports_unlocked() -> None:
    monitor = sensors_module.LockMonitor(run=runner_for(0, "(false,)"))
    assert monitor.locked() is False


def test_lock_failure_is_unknown_not_unlocked() -> None:
    monitor = sensors_module.LockMonitor(run=runner_for(1, ""))
    assert monitor.locked() is None


def test_lock_garbage_is_unknown() -> None:
    monitor = sensors_module.LockMonitor(run=runner_for(0, "Error: no such interface"))
    assert monitor.locked() is None


def test_battery_reports_on_battery() -> None:
    monitor = sensors_module.PowerMonitor(run=runner_for(0, "(<true>,)"))
    assert monitor.on_battery() is True


def test_battery_reports_on_ac() -> None:
    monitor = sensors_module.PowerMonitor(run=runner_for(0, "(<false>,)"))
    assert monitor.on_battery() is False


def test_battery_failure_is_unknown() -> None:
    monitor = sensors_module.PowerMonitor(run=runner_for(127, ""))
    assert monitor.on_battery() is None


def test_missing_gdbus_is_unknown() -> None:
    def missing(argv):
        raise OSError("no such file")

    assert sensors_module.LockMonitor(run=missing).locked() is None
    assert sensors_module.PowerMonitor(run=missing).on_battery() is None
