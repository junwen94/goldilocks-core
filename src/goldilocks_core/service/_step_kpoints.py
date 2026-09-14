"""Phase 3a of the v2 orchestrator (v2 epic 8, #8): the per-step
electronic-structure advisors (occupations/k-mesh/bands/convergence),
one of two ``_step*`` files ``_step.py`` composes into ``StepAdvice`` --
see that module's own docstring on why the per-step advisor set is
split across two files (the same import-surface ceiling reason
``_pseudo.py`` was split out of ``_system.py`` for).

Ordering follows the real dependency chain (each function's own
signature, not a convention invented here): ``occupations`` before
``k_sampling`` before ``n_irr_k``; ``nbnd`` needs
``system.electron_count``'s *plain* ``nelec`` (already resolved by
phase 2) so it is ``Blocked`` outright if that failed.
"""

from __future__ import annotations

from dataclasses import dataclass

from pymatgen.core import Structure

from goldilocks_core.advisors.convergence import (
    ConvergenceDecision,
    ConvergenceHumanInput,
    ConvergenceLlmInput,
    convergence,
)
from goldilocks_core.advisors.k_sampling import (
    KSamplingDecision,
    KSamplingHumanInput,
    KSamplingLlmInput,
    k_sampling,
)
from goldilocks_core.advisors.n_irr_k import NIrrKHumanInput, n_irr_k
from goldilocks_core.advisors.nbnd import (
    NbndDecision,
    NbndHumanInput,
    NbndLlmInput,
    nbnd,
)
from goldilocks_core.advisors.occupations import (
    OccupationsDecision,
    OccupationsHumanInput,
    OccupationsLlmInput,
    occupations,
)
from goldilocks_core.resolution import Blocked, FieldState, blocked_by
from goldilocks_core.service._analysis import AnalysisFacts
from goldilocks_core.service._system import SystemAdvice


@dataclass(frozen=True, slots=True)
class KpointsOverrides:
    occupations: OccupationsHumanInput | None = None
    occupations_llm: OccupationsLlmInput | None = None
    k_sampling: KSamplingHumanInput | None = None
    k_sampling_llm: KSamplingLlmInput | None = None
    n_irr_k: NIrrKHumanInput | None = None
    nbnd: NbndHumanInput | None = None
    nbnd_llm: NbndLlmInput | None = None
    convergence: ConvergenceHumanInput | None = None
    convergence_llm: ConvergenceLlmInput | None = None


@dataclass(frozen=True, slots=True)
class ElectronicStepAdvice:
    occupations: FieldState[OccupationsDecision]
    k_sampling: FieldState[KSamplingDecision]
    n_irr_k: FieldState[int]
    nbnd: FieldState[NbndDecision]
    convergence: FieldState[ConvergenceDecision]

    def field_states(self) -> tuple[FieldState[object], ...]:
        """Everything except ``occupations`` -- ``checks.check_all``
        folds that in itself; see its own docstring on why a caller
        must not repeat it here."""
        return (self.k_sampling, self.n_irr_k, self.nbnd, self.convergence)

    def records(self) -> dict[str, FieldState[object]]:
        return {
            "occupations": self.occupations,
            "k_sampling": self.k_sampling,
            "n_irr_k": self.n_irr_k,
            "nbnd": self.nbnd,
            "convergence": self.convergence,
        }


def electronic_step_advice(
    structure: Structure,
    analysis: AnalysisFacts,
    system: SystemAdvice,
    overrides: KpointsOverrides,
) -> ElectronicStepAdvice:
    occupations_state = occupations(
        analysis.is_metal,
        system.magnetic,
        overrides.occupations,
        overrides.occupations_llm,
    )
    k_sampling_state = k_sampling(
        analysis.is_metal,
        structure,
        occupations_state,
        overrides.k_sampling,
        overrides.k_sampling_llm,
    )
    n_irr_k_state = n_irr_k(structure, k_sampling_state, overrides.n_irr_k)

    if system.electron_count.ok:
        nbnd_state = nbnd(
            system.electron_count.value.nelec,
            occupations_state,
            "scf",
            system.magnetic,
            overrides.nbnd,
            overrides.nbnd_llm,
        )
    else:
        nbnd_state = Blocked(by=blocked_by(system.electron_count))

    convergence_state = convergence(
        len(structure),
        analysis.needs_correlation,
        analysis.geometry,
        overrides.convergence,
        overrides.convergence_llm,
    )

    return ElectronicStepAdvice(
        occupations=occupations_state,
        k_sampling=k_sampling_state,
        n_irr_k=n_irr_k_state,
        nbnd=nbnd_state,
        convergence=convergence_state,
    )
