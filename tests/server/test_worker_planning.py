from __future__ import annotations

import multiprocessing
import os
import sys
import time
from multiprocessing.connection import wait

import pytest

from goldilocks_core.server import workers

GIB = 1024**3


def test_plan_workers_single_cpu_returns_one() -> None:
    assert workers.plan_workers(1, 100 * GIB, GIB) == 1


def test_plan_workers_without_measured_quantities_uses_cpu_count() -> None:
    assert workers.plan_workers(8, None, None) == 8
    assert workers.plan_workers(8, None, GIB) == 8
    assert workers.plan_workers(8, 100 * GIB, None) == 8


def test_plan_workers_caps_by_memory_budget() -> None:
    assert workers.plan_workers(8, 2 * GIB, GIB) == 1
    assert workers.plan_workers(8, 5 * GIB, 2 * GIB) == 1
    assert workers.plan_workers(8, 10 * GIB, 2 * GIB) == 3


def test_plan_workers_never_exceeds_cpu_count() -> None:
    assert workers.plan_workers(4, 1000 * GIB, 100 * GIB) == 4


def test_plan_workers_keeps_at_least_one_worker() -> None:
    assert workers.plan_workers(8, GIB, 1000 * GIB) == 1


def test_plan_workers_ignores_nonpositive_cost() -> None:
    assert workers.plan_workers(8, 2 * GIB, 0) == 8


def test_cgroup_v2_cpu_quota_converts_to_cpus(tmp_path) -> None:
    (tmp_path / "cpu.max").write_text("200000 100000\n", encoding="utf-8")
    assert workers._cgroup_cpu_quota(tmp_path) == 2


def test_cgroup_v2_cpu_quota_rounds_partial_cpus_up(tmp_path) -> None:
    (tmp_path / "cpu.max").write_text("150000 100000\n", encoding="utf-8")
    assert workers._cgroup_cpu_quota(tmp_path) == 2


def test_cgroup_v2_unlimited_cpu_quota_is_no_limit(tmp_path) -> None:
    (tmp_path / "cpu.max").write_text("max 100000\n", encoding="utf-8")
    assert workers._cgroup_cpu_quota(tmp_path) is None


def test_cgroup_v1_cpu_quota_converts_to_cpus(tmp_path) -> None:
    (tmp_path / "cpu").mkdir()
    (tmp_path / "cpu" / "cpu.cfs_quota_us").write_text("300000\n", encoding="utf-8")
    (tmp_path / "cpu" / "cpu.cfs_period_us").write_text("100000\n", encoding="utf-8")
    assert workers._cgroup_cpu_quota(tmp_path) == 3


def test_cgroup_v1_disabled_quota_is_no_limit(tmp_path) -> None:
    (tmp_path / "cpu").mkdir()
    (tmp_path / "cpu" / "cpu.cfs_quota_us").write_text("-1\n", encoding="utf-8")
    (tmp_path / "cpu" / "cpu.cfs_period_us").write_text("100000\n", encoding="utf-8")
    assert workers._cgroup_cpu_quota(tmp_path) is None


def test_cgroup_memory_limit_placeholder_is_no_limit(tmp_path) -> None:
    (tmp_path / "memory.max").write_text("max\n", encoding="utf-8")
    assert workers._cgroup_memory_limit_bytes(tmp_path) is None


def test_cgroup_memory_limit_reads_bytes(tmp_path) -> None:
    (tmp_path / "memory.max").write_text("4294967296\n", encoding="utf-8")
    assert workers._cgroup_memory_limit_bytes(tmp_path) == 4 * GIB


def test_cgroup_v1_huge_limit_is_no_limit(tmp_path) -> None:
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "memory.limit_in_bytes").write_text(
        f"{1 << 63}\n", encoding="utf-8"
    )
    assert workers._cgroup_memory_limit_bytes(tmp_path) is None


def test_memory_budget_takes_the_smallest_limit(tmp_path, monkeypatch) -> None:
    (tmp_path / "memory.max").write_text(f"{4 * GIB}\n", encoding="utf-8")
    monkeypatch.setattr(
        workers, "_available_memory_bytes", lambda: 100 * GIB, raising=True
    )
    assert workers.memory_budget_bytes(tmp_path) == 4 * GIB


def test_default_workers_env_override_wins_before_measurement(monkeypatch) -> None:
    monkeypatch.setenv(workers.WORKERS_ENV, "3")
    assert workers.default_workers() == 3


def test_default_workers_rejects_nonnumeric_override(monkeypatch) -> None:
    monkeypatch.setenv(workers.WORKERS_ENV, "two")
    with pytest.raises(ValueError):
        workers.default_workers()


def test_default_workers_pins_to_at_least_one(monkeypatch) -> None:
    monkeypatch.setenv(workers.WORKERS_ENV, "0")
    assert workers.default_workers() == 1


def _no_op() -> None:
    pass


def _report_alive(sender, diagnostic: bool) -> None:
    if diagnostic:
        sender.send("starting")
    workers._die_with_master()
    sender.send("alive")
    sender.close()


def test_die_with_master_keeps_worker_whose_master_is_alive(monkeypatch) -> None:
    """A live master that is PID 1 (the Docker default) is not reparenting."""
    monkeypatch.setenv(workers.MASTER_PID_ENV, str(os.getpid()))
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(False)
    process = context.Process(target=_report_alive, args=(sender, True))
    process.start()
    sender.close()
    assert receiver.poll(timeout=30)
    assert receiver.recv() == "starting"
    assert receiver.recv() == "alive"
    process.join(timeout=30)
    assert process.exitcode == 0


def test_die_with_master_exits_when_reparented(monkeypatch) -> None:
    """A worker whose recorded master is gone kills itself."""
    context = multiprocessing.get_context("spawn")
    gone = context.Process(target=_no_op)
    gone.start()
    gone.join(timeout=30)
    monkeypatch.setenv(workers.MASTER_PID_ENV, str(gone.pid))
    receiver, sender = context.Pipe(False)
    process = context.Process(target=_report_alive, args=(sender, False))
    process.start()
    sender.close()
    assert wait([receiver], timeout=30)
    with pytest.raises(EOFError):
        receiver.recv()
    process.join(timeout=30)
    assert process.exitcode == 0


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX reparenting check")
def test_master_alive_rejects_a_non_parent_pid() -> None:
    assert workers._master_alive(os.getpid()) is False


def test_die_with_master_leaves_the_master_alone(monkeypatch) -> None:
    """At one worker the master runs the factory itself and must survive."""
    monkeypatch.setenv(workers.MASTER_PID_ENV, str(os.getpid()))
    workers._die_with_master()


@pytest.mark.skipif(sys.platform == "win32", reason="fork-based cost measurement")
def test_measured_worker_cost_propagates_child_failure(monkeypatch) -> None:
    class ExplodingBackend:
        def prewarm(self) -> None:
            raise RuntimeError("model exploded")

    monkeypatch.setattr("goldilocks_core.advice.kdistance.QrfBackend", ExplodingBackend)
    with pytest.raises(RuntimeError, match="model exploded"):
        workers.measured_worker_cost_bytes()


def test_measured_worker_cost_drops_out_when_child_hangs(monkeypatch) -> None:
    class SlowBackend:
        def prewarm(self) -> None:
            time.sleep(5)

    monkeypatch.setattr("goldilocks_core.advice.kdistance.QrfBackend", SlowBackend)
    monkeypatch.setattr(workers, "MEASURE_TIMEOUT_SECONDS", 0.2)
    assert workers.measured_worker_cost_bytes() is None
