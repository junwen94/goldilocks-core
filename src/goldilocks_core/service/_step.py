"""``StepAdvice``/``StepOverrides``: composes the two per-step-advisor
phases -- ``_step_kpoints.py`` (occupations/k-mesh/bands/convergence)
and ``_step_resources.py`` (job/parallelisation) -- one level up,
mirroring ``_advice.Advice``'s own composition of
``AnalysisFacts``/``SystemAdvice``/``StepAdvice`` (v2 epic 8, #8).

Split into two advisor phases, and this composing file kept separate
from both: a flat ``StepAdvice`` importing all eight per-step advisor
modules' types directly (needed just for its own field annotations)
would exceed this project's import-surface ceiling
(``scripts/check_complexity.py``, default 12 origins/24 symbols) on its
own, the same problem ``_advice.py`` solves for the top-level ``Advice``.
Composing two already-typed sub-objects costs this module only two
local origins.

``steps_advice`` builds every pw.x step of one task together (#169):
each step's electronic advice first, then the task's resources over all
of them, since the steps share one job (``_step_resources.py``).

``StepOverrides`` is one step's request overrides (#154). A request is
routed per step by ``set_overrides.build_overrides``, so each field holds
only settings that apply to its step: ``kpoints``/``resources`` on a pw.x
step, ``relax`` on a relax or vc-relax step, ``dos`` on the dos.x step.
A setting given without a step that lands on more than one step raises
``applies_to_multiple_steps``, declared here beside the bags it fills.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from goldilocks_core.advisors.dos import DosHumanInput, DosLlmInput
from goldilocks_core.advisors.job_resources import JobHumanInput, JobLlmInput
from goldilocks_core.advisors.relax import RelaxHumanInput, RelaxLlmInput
from goldilocks_core.inputs.hpc import HpcProfile
from goldilocks_core.resolution import FieldState, Warning
from goldilocks_core.service._analysis import AnalysisFacts
from goldilocks_core.service._step_kpoints import (
    ElectronicStepAdvice,
    KpointsOverrides,
    electronic_step_advice,
)
from goldilocks_core.service._step_protocol import StepProtocol
from goldilocks_core.service._step_resources import (
    ResourceOverrides,
    ResourceStepAdvice,
    task_resources,
)
from goldilocks_core.service._system import SystemAdvice

_APPLIES_TO_MULTIPLE_STEPS = "overrides.applies_to_multiple_steps"

OVERRIDES_WARNING_CATALOGUE: tuple[Warning, ...] = (
    Warning(
        code=_APPLIES_TO_MULTIPLE_STEPS,
        level="info",
        category="overrides",
        message=(
            "A setting given without steps.<step>. applies to every step of "
            "the task it fits; this notice names them when that is more than one."
        ),
    ),
)
"""The routing notices ``capabilities`` publishes, beside the advisors'."""


def applies_to_multiple_steps(key: str, steps: tuple[str, ...]) -> Warning:
    """The notice that ``key``, given without a step, lands on ``steps``."""
    return Warning(
        code=_APPLIES_TO_MULTIPLE_STEPS,
        level="info",
        category="overrides",
        message=(
            f"{key} was given without a step, so it applies to the "
            f"{', '.join(steps[:-1])} and {steps[-1]} steps; set "
            f"steps.<step>.{key} to set one step only."
        ),
    )


@dataclass(frozen=True, slots=True)
class RelaxOverrides:
    """A relax or vc-relax step's own settings (v2 epic 10, #10).
    ``capabilities.py``'s ``_leaves()`` walks this dataclass the same way it
    walks ``KpointsOverrides``/``ResourceOverrides`` -- see that module's own
    docstring."""

    relax: RelaxHumanInput | None = None
    relax_llm: RelaxLlmInput | None = None


@dataclass(frozen=True, slots=True)
class DosOverrides:
    """The dos.x step's own settings (``advisors/dos.py``), reachable from a
    request since #154."""

    dos: DosHumanInput | None = None
    dos_llm: DosLlmInput | None = None


@dataclass(frozen=True, slots=True)
class StepOverrides:
    """One step's request overrides (module docstring)."""

    kpoints: KpointsOverrides = field(default_factory=KpointsOverrides)
    resources: ResourceOverrides = field(default_factory=ResourceOverrides)
    relax: RelaxOverrides = field(default_factory=RelaxOverrides)
    dos: DosOverrides = field(default_factory=DosOverrides)


@dataclass(frozen=True, slots=True)
class StepAdvice:
    kpoints: ElectronicStepAdvice
    resources: ResourceStepAdvice

    def field_states(self) -> tuple[FieldState[object], ...]:
        """Everything except ``occupations``/``nbnd`` --
        ``checks.check_all`` folds those in itself; see
        ``ElectronicStepAdvice``'s own docstring on why a caller must not
        repeat them here."""
        return self.kpoints.field_states() + self.resources.field_states()

    def records(self) -> dict[str, FieldState[object]]:
        return {**self.kpoints.records(), **self.resources.records()}


def steps_advice(
    analysis: AnalysisFacts,
    system: SystemAdvice,
    hpc: HpcProfile,
    steps: Mapping[str, StepOverrides],
    job_human: JobHumanInput | None,
    job_llm: JobLlmInput | None,
    protocols: Mapping[str, StepProtocol],
) -> dict[str, StepAdvice]:
    """One ``StepAdvice`` per entry of ``steps`` (a pw.x step's purpose ->
    its overrides, in run order), all on the simulated cell
    ``system.cell``. The steps share one job, sized with the task-level
    ``job_human``/``job_llm``; each step's own overrides give its
    k-point/band advice and npool/ndiag, over its task's protocol for it
    in ``protocols`` (by purpose; a step without one has none, #174), and
    its purpose words its band count's moment warning (#118)."""
    electronic = {
        purpose: electronic_step_advice(
            analysis,
            system,
            step.kpoints,
            protocols.get(purpose, StepProtocol()),
            purpose,
        )
        for purpose, step in steps.items()
    }
    resources = task_resources(
        system,
        electronic,
        hpc,
        job_human,
        job_llm,
        {purpose: step.resources.parallelisation for purpose, step in steps.items()},
    )
    return {
        purpose: StepAdvice(kpoints=electronic[purpose], resources=resources[purpose])
        for purpose in steps
    }
