"""``PlannedStep``/``expand_task``: one task becomes an ordered step
sequence.

New in v2 (v2 epic 9, #9). ``steps.py``'s own docstring named this gap
and deferred it deliberately: "the design doc's layer 1 (``PlannedStep``
+ ``expand_task``, goldilocks-core-design.md:1072-1083) exists to turn
one task into an *ordered sequence* of steps (e.g. ``dos -> [scf, nscf,
dos]``)... add ``PlannedStep``/``plan.py`` once a second task actually
needs step sequencing." ``dos`` (``types.CalcTask``, this epic) is that
second task.

Lives top-level next to ``steps.py``/``step_settings.py``, same
reasoning as both of those modules' own docstrings: used across
multiple packages (``service/``, ``generation/``), no single owner.

**Scope, per this epic's own acceptance bar** (goldilocks-implementation
-plan.md's five-scenario table, scenario 5 -- "DOS multi-step"): prove
the type-level split and a real multi-step path actually work end to
end, not full DOS science. ``expand_task`` only decides *how many steps,
in what order, running what program* -- never a scientific value (that
stays ``advisors/``'s job, per every other module in this codebase).
``CalcTask`` being an exhaustive ``Literal`` (``types.py``, this epic)
means the two branches below are provably complete -- no fallback/raise
branch is needed or added speculatively for a task that cannot exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from goldilocks_core.inputs.task import Task


@dataclass(frozen=True, slots=True)
class PlannedStep:
    """One step in a task's sequence, before any settings are resolved
    for it -- just *what* runs and *why* (``purpose``), not yet *how*.
    ``purpose`` is deliberately not ``types.CalcTask``: a multi-step task
    like ``dos`` has steps with purposes (``"scf"``, ``"nscf"``) that are
    not tasks in their own right, only ever ingredients of one. ``relax``/
    ``vc-relax`` (v2 epic 10, #10) *are* both a task and their own single
    step's purpose -- one ``pw.x`` run does the whole task."""

    purpose: Literal["scf", "nscf", "dos", "relax", "vc-relax"]
    program: Literal["pw.x", "dos.x"]


def expand_task(task: Task) -> tuple[PlannedStep, ...]:
    """``CalcTask`` is an exhaustive ``Literal`` (``types.py``), so these
    branches are provably complete -- no fallback/raise branch is needed
    or added speculatively for a task that cannot exist."""
    if task.task == "scf_single_point":
        return (PlannedStep(purpose="scf", program="pw.x"),)
    if task.task in ("relax", "vc-relax"):
        return (PlannedStep(purpose=task.task, program="pw.x"),)
    return (
        PlannedStep(purpose="scf", program="pw.x"),
        PlannedStep(purpose="nscf", program="pw.x"),
        PlannedStep(purpose="dos", program="dos.x"),
    )
