"""Phase 3b of the v2 orchestrator (v2 epic 8, #8): resource sizing for
a task, the other of the two ``_step*`` files ``_step.py`` composes into
``StepAdvice`` -- see that module's own docstring on why the per-step
advisor set is split across two files.

**One job per task, parallelisation per step.** A task's pw.x steps run
one after another in one allocation, so ``task_resources`` makes one
``job_resources`` call over every step's estimate (#169): the largest
step sizes the allocation, and each step gets its own ``max_seconds``
share of the walltime, which the step's ``ResourceStepAdvice`` carries
for ``generate()``. Each step's ``parallelisation`` then reads that
job's ``ntasks`` and the step's own k-point/spin pairs (``n_irr_k`` and
the magnetic decision's spin layout) -- ``job_resources`` never reads
``parallelisation``, only the reverse, by design (see
``advisors/parallelisation.py``'s own module note on false circularity).
``job``/``job_llm`` overrides are task-level; ``npool``/``ndiag`` are
per step.

A step's estimate needs ``cutoffs``/``nbnd``/``magnetic``/``n_irr_k``
resolved, so the job is ``Blocked`` by the first of those that is not,
in any step.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from pymatgen.core import Structure

from goldilocks_core.advisors.job_resources import (
    JobDecision,
    JobHumanInput,
    JobLlmInput,
    job_resources,
)
from goldilocks_core.advisors.parallelisation import (
    ParallelisationDecision,
    ParallelisationHumanInput,
    kpoint_spin_pairs,
    parallelisation,
)
from goldilocks_core.advisors.size import (
    ResourceEstimate,
    memory_per_process,
    resource_estimate,
)
from goldilocks_core.inputs.hpc import HpcProfile
from goldilocks_core.plan import SCF_LIKE_PURPOSES
from goldilocks_core.resolution import Blocked, FieldState, blocked_by
from goldilocks_core.service._step_kpoints import ElectronicStepAdvice
from goldilocks_core.service._system import SystemAdvice


@dataclass(frozen=True, slots=True)
class ResourceOverrides:
    job: JobHumanInput | None = None
    job_llm: JobLlmInput | None = None
    parallelisation: ParallelisationHumanInput | None = None


@dataclass(frozen=True, slots=True)
class ResourceStepAdvice:
    """One step's resources: the task's one ``job`` (the same state in
    every step of the task), the step's own ``parallelisation``, and its
    ``max_seconds`` share of the job's walltime
    (``JobDecision.max_seconds_by_step``), ``None`` while the job is not
    resolved."""

    job: FieldState[JobDecision]
    parallelisation: FieldState[ParallelisationDecision]
    max_seconds: int | None
    resource_estimate: ResourceEstimate | None = None
    mem_per_proc: float | None = None

    def field_states(self) -> tuple[FieldState[object], ...]:
        return (self.job, self.parallelisation)

    def records(self) -> dict[str, FieldState[object]]:
        return {"job": self.job, "parallelisation": self.parallelisation}


def task_resources(
    structure: Structure,
    system: SystemAdvice,
    electronic: Mapping[str, ElectronicStepAdvice],
    hpc: HpcProfile,
    job_human: JobHumanInput | None,
    job_llm: JobLlmInput | None,
    parallelisation_human: Mapping[str, ParallelisationHumanInput | None],
) -> dict[str, ResourceStepAdvice]:
    """Each step's resources, keyed like ``electronic`` (purpose -> the
    step's electronic advice): one job from ``job_human``/``job_llm``,
    and each step's npool/ndiag from ``parallelisation_human[purpose]``."""
    unresolved = next(
        (
            state
            for step in electronic.values()
            for state in (system.cutoffs, step.nbnd, system.magnetic, step.n_irr_k)
            if not state.ok
        ),
        None,
    )
    if unresolved is not None:
        blocked = Blocked(by=blocked_by(unresolved))
        return {
            purpose: ResourceStepAdvice(
                job=blocked, parallelisation=Blocked(by=blocked), max_seconds=None
            )
            for purpose in electronic
        }

    pairs = {
        purpose: kpoint_spin_pairs(step.n_irr_k.value, system.magnetic.value)
        for purpose, step in electronic.items()
    }
    estimates = {
        purpose: _step_estimate(
            structure,
            system,
            step.nbnd.value.nbnd,
            pairs[purpose],
            scf_loop=purpose in SCF_LIKE_PURPOSES,
        )
        for purpose, step in electronic.items()
    }
    job = job_resources(estimates, hpc, job_human, job_llm)
    return {
        purpose: _step_resources(
            job,
            purpose,
            estimates[purpose],
            pairs[purpose],
            parallelisation_human[purpose],
        )
        for purpose in electronic
    }


def _step_estimate(
    structure: Structure,
    system: SystemAdvice,
    nbnd: int,
    n_kpoint_spin: int,
    *,
    scf_loop: bool,
) -> ResourceEstimate:
    magnetic = system.magnetic.value
    return resource_estimate(
        structure,
        system.cutoffs.value.ecutwfc_ry,
        system.cutoffs.value.ecutrho_ry,
        nbnd,
        n_kpoint_spin,
        scf_loop,
        nspin=2 if magnetic.spin_polarized else 1,
        noncollinear=magnetic.spin_orbit_enabled,
    )


def _step_resources(
    job: FieldState[JobDecision],
    purpose: str,
    estimate: ResourceEstimate,
    n_kpoint_spin: int,
    human: ParallelisationHumanInput | None,
) -> ResourceStepAdvice:
    if not job.ok:
        return ResourceStepAdvice(
            job=job,
            parallelisation=Blocked(by=blocked_by(job)),
            max_seconds=None,
            resource_estimate=estimate,
        )
    parallel = parallelisation(job.value.ntasks, n_kpoint_spin, human)
    return ResourceStepAdvice(
        job=job,
        parallelisation=parallel,
        max_seconds=job.value.max_seconds_by_step[purpose],
        resource_estimate=estimate,
        mem_per_proc=memory_per_process(
            estimate, job.value.ntasks, parallel.value.npool
        ),
    )
