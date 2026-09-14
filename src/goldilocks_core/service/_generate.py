"""``generate()``: the "delivery" half of the v2 orchestrator, unwrapping
one ``Advice`` into rendered ``Step``s (v2 epic 8, #8). See
``_pipeline.py`` for the ``advise()``/``generate()`` promise this splits.
"""

from __future__ import annotations

from goldilocks_core.checks import CheckReport
from goldilocks_core.failures import ExpectedFailure
from goldilocks_core.generation.quantum_espresso.scf import write_qe_scf
from goldilocks_core.service._advice import Advice
from goldilocks_core.step_settings import PwSettings
from goldilocks_core.steps import SharedContext, Step, default_shared_context
from goldilocks_core.system_settings import SystemSettings


class AdviceIncomplete(ExpectedFailure, ValueError):
    """``generate()`` was asked to render a runnable input, but
    ``checks.check_all`` found at least one required field
    ``unavailable``/``blocked``. Carries the ``CheckReport`` so a caller
    can render the blocking reasons without re-deriving them."""

    kind = "advice_incomplete"

    def __init__(self, report: CheckReport) -> None:
        self.report = report
        # report.blocking has one entry per blocked field, not per
        # distinct root cause -- many fields sharing one upstream
        # failure (e.g. every pseudopotential-dependent field, once
        # pseudo_table itself is Unavailable) would otherwise repeat
        # the same sentence once per field. dict.fromkeys dedupes while
        # keeping the first-seen order, which collect_blocked's own
        # order already is.
        reasons = dict.fromkeys(report.blocking)
        super().__init__("cannot generate a runnable input: " + "; ".join(reasons))


def generate(
    advice: Advice, report: CheckReport, *, ctx: SharedContext | None = None
) -> tuple[Step, ...]:
    """Delivery: raises ``AdviceIncomplete`` if ``report`` is not ``ok``,
    per the "generate() refuses iff a needed field is unavailable/blocked"
    promise -- callers must call ``check(advice)`` first and are expected
    to inspect ``report.blocking`` themselves before ever reaching here
    in normal operation; this raise is the hard boundary, not the
    primary way a caller learns what is missing."""
    if not report.ok:
        raise AdviceIncomplete(report)
    system = SystemSettings(
        functional=advice.system.functional.value,
        cutoffs=advice.system.cutoffs.value,
        electron_count=advice.system.electron_count.value,
        pseudopotentials=advice.system.pseudo.metadata.value,
        magnetic=advice.system.magnetic.value,
        vdw=advice.system.vdw.value,
        hubbard=advice.system.hubbard.value,
        boundary=advice.system.boundary.value,
    )
    kpoints = advice.step.kpoints
    step_settings = PwSettings(
        disk_io=None,
        size=advice.step.resource_estimate,
        mem_per_proc=advice.step.mem_per_proc,
        occupations=kpoints.occupations.value,
        k_sampling=kpoints.k_sampling.value,
        n_irr_k=kpoints.n_irr_k.value if kpoints.n_irr_k.ok else None,
        nbnd=kpoints.nbnd.value,
        convergence=kpoints.convergence.value,
        parallel=advice.step.resources.parallelisation.value,
        relax=None,
    )
    return tuple(
        write_qe_scf(
            system,
            step_settings,
            advice.step.resources.job.value,
            ctx or default_shared_context(),
        )
    )
