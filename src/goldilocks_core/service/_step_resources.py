"""Phase 3b of the v2 orchestrator (v2 epic 8, #8): per-step resource
sizing (envelope/job/parallelisation), the other of the two ``_step*``
files ``_step.py`` composes into ``StepAdvice`` -- see that module's own
docstring on why the per-step advisor set is split across two files.

Takes ``nbnd``/``n_irr_k`` as plain ``FieldState`` parameters rather
than the whole ``ElectronicStepAdvice`` object, so this module never
needs to import ``_step_kpoints`` at all: the resource-sizing chain
only ever reads those two fields from phase 3a's output.

The resource envelope needs ``cutoffs``/``nbnd``/``magnetic`` all
resolved before ``job_resources`` can run, and ``parallelisation`` needs
``job_resources``'s plain output -- ``job_resources`` never reads
``parallelisation``, only the reverse, by design (see
``advisors/parallelisation.py``'s own module note on false circularity).
"""

from __future__ import annotations

from dataclasses import dataclass

from pymatgen.core import Structure

from goldilocks_core.advisors.job_resources import (
    JobDecision,
    JobHumanInput,
    JobLlmInput,
    job_resources,
)
from goldilocks_core.advisors.nbnd import NbndDecision
from goldilocks_core.advisors.parallelisation import (
    ParallelisationDecision,
    ParallelisationHumanInput,
    parallelisation,
)
from goldilocks_core.advisors.size import (
    ResourceEstimate,
    envelope,
    memory_per_process,
    resource_estimate,
)
from goldilocks_core.inputs.hpc import HpcProfile
from goldilocks_core.resolution import Blocked, FieldState, blocked_by
from goldilocks_core.service._system import SystemAdvice


@dataclass(frozen=True, slots=True)
class ResourceOverrides:
    job: JobHumanInput | None = None
    job_llm: JobLlmInput | None = None
    parallelisation: ParallelisationHumanInput | None = None


@dataclass(frozen=True, slots=True)
class ResourceStepAdvice:
    job: FieldState[JobDecision]
    parallelisation: FieldState[ParallelisationDecision]
    resource_estimate: ResourceEstimate | None = None
    mem_per_proc: float | None = None

    def field_states(self) -> tuple[FieldState[object], ...]:
        return (self.job, self.parallelisation)

    def records(self) -> dict[str, FieldState[object]]:
        return {"job": self.job, "parallelisation": self.parallelisation}


def resource_step_advice(
    structure: Structure,
    code: str,
    system: SystemAdvice,
    nbnd: FieldState[NbndDecision],
    n_irr_k: FieldState[int],
    hpc: HpcProfile,
    overrides: ResourceOverrides,
) -> ResourceStepAdvice:
    estimate: ResourceEstimate | None = None
    job_state: FieldState[JobDecision]
    if system.cutoffs.ok and nbnd.ok and system.magnetic.ok:
        estimate = envelope(
            [
                resource_estimate(
                    structure,
                    system.cutoffs.value.ecutwfc_ry,
                    system.cutoffs.value.ecutrho_ry,
                    nbnd.value.nbnd,
                    nspin=2 if system.magnetic.value.spin_polarized else 1,
                    noncollinear=system.magnetic.value.spin_orbit_enabled,
                )
            ]
        )
        job_state = job_resources(estimate, hpc, overrides.job, overrides.job_llm)
    else:
        blocker = next(
            state for state in (system.cutoffs, nbnd, system.magnetic) if not state.ok
        )
        job_state = Blocked(by=blocked_by(blocker))

    mem_per_proc: float | None = None
    parallel_state: FieldState[ParallelisationDecision]
    if job_state.ok:
        has_scalapack = hpc.has_scalapack.get(code, False)
        n_irr_k_value = n_irr_k.value if n_irr_k.ok else None
        parallel_state = parallelisation(
            job_state.value, has_scalapack, n_irr_k_value, overrides.parallelisation
        )
        if parallel_state.ok and estimate is not None:
            mem_per_proc = memory_per_process(
                estimate, job_state.value.ntasks, parallel_state.value.npool
            )
    else:
        parallel_state = Blocked(by=blocked_by(job_state))

    return ResourceStepAdvice(
        job=job_state,
        parallelisation=parallel_state,
        resource_estimate=estimate,
        mem_per_proc=mem_per_proc,
    )
