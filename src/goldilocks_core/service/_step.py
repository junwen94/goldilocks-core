"""``StepAdvice``/``StepOverrides``: composes the two per-step-advisor
phases -- ``_step_kpoints.py`` (occupations/k-mesh/bands/convergence)
and ``_step_resources.py`` (resource envelope/job/parallelisation) --
one level up, mirroring ``_advice.Advice``'s own composition of
``AnalysisFacts``/``SystemAdvice``/``StepAdvice`` (v2 epic 8, #8).

Split into two advisor phases, and this composing file kept separate
from both: a flat ``StepAdvice`` importing all eight per-step advisor
modules' types directly (needed just for its own field annotations)
would exceed this project's import-surface ceiling
(``scripts/check_complexity.py``, default 12 origins/24 symbols) on its
own, the same problem ``_advice.py`` solves for the top-level ``Advice``.
Composing two already-typed sub-objects costs this module only two
local origins.

Only the one task epics 1-7 actually built a generation writer for
(``scf_single_point`` on Quantum ESPRESSO) is wired here. Task
sequencing (``PlannedStep``/``plan.py``) does not exist yet either
(``steps.py``'s own docstring: "add once a second task actually needs
step sequencing").
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pymatgen.core import Structure

from goldilocks_core.advisors.size import ResourceEstimate
from goldilocks_core.inputs.hpc import HpcProfile
from goldilocks_core.resolution import FieldState
from goldilocks_core.service._analysis import AnalysisFacts
from goldilocks_core.service._step_kpoints import (
    ElectronicStepAdvice,
    KpointsOverrides,
    electronic_step_advice,
)
from goldilocks_core.service._step_resources import (
    ResourceOverrides,
    ResourceStepAdvice,
    resource_step_advice,
)
from goldilocks_core.service._system import SystemAdvice


@dataclass(frozen=True, slots=True)
class StepOverrides:
    kpoints: KpointsOverrides = field(default_factory=KpointsOverrides)
    resources: ResourceOverrides = field(default_factory=ResourceOverrides)


@dataclass(frozen=True, slots=True)
class StepAdvice:
    kpoints: ElectronicStepAdvice
    resources: ResourceStepAdvice

    @property
    def resource_estimate(self) -> ResourceEstimate | None:
        return self.resources.resource_estimate

    @property
    def mem_per_proc(self) -> float | None:
        return self.resources.mem_per_proc

    def field_states(self) -> tuple[FieldState[object], ...]:
        """Everything except ``occupations`` -- ``checks.check_all``
        folds that in itself; see ``ElectronicStepAdvice``'s own
        docstring on why a caller must not repeat it here."""
        return self.kpoints.field_states() + self.resources.field_states()

    def records(self) -> dict[str, FieldState[object]]:
        return {**self.kpoints.records(), **self.resources.records()}


def step_advice(
    structure: Structure,
    code: str,
    analysis: AnalysisFacts,
    system: SystemAdvice,
    hpc: HpcProfile,
    overrides: StepOverrides,
) -> StepAdvice:
    kpoints = electronic_step_advice(structure, analysis, system, overrides.kpoints)
    resources = resource_step_advice(
        structure, code, system, kpoints.nbnd, kpoints.n_irr_k, hpc, overrides.resources
    )
    return StepAdvice(kpoints=kpoints, resources=resources)
