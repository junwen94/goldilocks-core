"""``Advice``/``RunOverrides``: the pipeline-wide shapes composing the
three phases in ``_analysis.py``/``_system.py``/``_step.py`` (v2 epic 8,
#8).

Composed of the three phases' own dataclasses, not a flat 20-odd-field
dataclass: a flat shape would need to import every analysis/advisor
decision type directly for its own field annotations, which is exactly
the "god module" shape ``scripts/check_complexity.py``'s import-surface
ceiling exists to catch. Composing three already-typed sub-objects, plus
the task's ``JobOverrides`` (``_step_resources.py``) in ``RunOverrides``,
costs this module only four local origins.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from goldilocks_core.resolution import FieldState, Warning, collect_warnings
from goldilocks_core.service._analysis import AnalysisFacts, AnalysisOverrides
from goldilocks_core.service._step import StepAdvice, StepOverrides
from goldilocks_core.service._step_resources import JobOverrides
from goldilocks_core.service._system import SystemAdvice, SystemOverrides


@dataclass(frozen=True, slots=True)
class RunOverrides:
    """One task's request overrides, routed for that task
    (``set_overrides.build_overrides``, #154): the analysis and system
    overrides and the task's one ``job`` allocation, then ``steps``, one
    ``StepOverrides`` per step of the task that the request set anything
    for, keyed by the step's purpose (``plan.expand_task``). Typed, not a
    generic key/value bag -- each advisor's override class is separate.

    ``warnings`` holds the routing notices (a setting given without a step
    that landed on several); only a task with two pw.x steps, ``dos``, can
    raise one, so ``DosAdvice`` alone publishes them."""

    analysis: AnalysisOverrides = field(default_factory=AnalysisOverrides)
    system: SystemOverrides = field(default_factory=SystemOverrides)
    job: JobOverrides = field(default_factory=JobOverrides)
    steps: Mapping[str, StepOverrides] = field(default_factory=dict)
    warnings: tuple[Warning, ...] = ()

    def for_steps(self, purposes: Sequence[str]) -> dict[str, StepOverrides]:
        """Each of ``purposes``' (every step of the task, in run order)
        overrides, empty where the request set none. Raises ``ValueError``
        when ``steps`` holds a step outside ``purposes``: these overrides
        were routed for another task."""
        foreign = [purpose for purpose in self.steps if purpose not in purposes]
        if foreign:
            raise ValueError(
                f"overrides for step(s) {', '.join(foreign)} were routed for "
                f"another task; this task's steps are {', '.join(purposes)}"
            )
        return {
            purpose: self.steps.get(purpose, StepOverrides()) for purpose in purposes
        }


@dataclass(frozen=True, slots=True)
class Advice:
    """Every ``FieldState`` produced for one structure/code/task request.
    ``checks.check_all(*advice.field_states(), ...)`` decides whether
    ``generate()`` may proceed; ``advice.records()`` is what
    ``BundleInput.records`` should carry unchanged -- tri-state, not the
    unwrapped values ``SystemSettings``/``PwSettings`` need."""

    analysis: AnalysisFacts
    system: SystemAdvice
    step: StepAdvice

    def field_states(self) -> tuple[FieldState[object], ...]:
        """Every ``FieldState`` except ``occupations``/``functional``/
        ``magnetic``/``electron_count``/``vdw``/``nbnd``/``boundary``, which
        ``checks.check_all`` folds in itself -- see its own docstring on why
        a caller must not repeat them here."""
        return (
            self.analysis.field_states()
            + self.system.field_states()
            + self.step.field_states()
        )

    def records(self) -> dict[str, FieldState[object]]:
        """One entry per analysis/advisor decision, keyed by the same
        name used throughout this package -- the manifest's ``records``
        object. No such key set is codified anywhere else in the
        codebase; this is epic 8's own convention.

        The values are the advisors' own typed ``FieldState``s, as
        ``RelaxAdvice``/``DosAdvice`` add theirs: they are projected to
        JSON once, at the boundary (``resolution.records_document``), so
        no record reaches an output by a second path (#152)."""
        return {
            **self.analysis.records(),
            **self.system.records(),
            **self.step.records(),
        }

    def warnings(self) -> list[dict[str, object]]:
        """Every record's own ``resolution.Warning``s, flattened into one
        machine-actionable list (``resolution.collect_warnings``), each
        tagged with the ``record`` it came from -- goldilocks-agent
        -design.md's "tool return value must carry a warnings array, the
        agent must relay it verbatim" requirement, as plain dicts a caller
        can filter/group by ``code``/``category``/``record`` without
        parsing text."""
        return collect_warnings(self.records())
