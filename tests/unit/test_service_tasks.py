"""``service.TASK_PHASES``: the one table every transport looks a task's
service functions up in (#118)."""

from __future__ import annotations

from typing import get_args

from goldilocks_core.service import TASK_PHASES
from goldilocks_core.types import CalcTask


def test_every_task_has_exactly_one_row() -> None:
    """The CLI, HTTP and MCP ``explain``/``run`` dispatch through this
    table, so a task without a row raises ``KeyError`` instead of running
    another task's functions, as their old ``else`` branches ran scf."""
    assert TASK_PHASES.keys() == set(get_args(CalcTask))
