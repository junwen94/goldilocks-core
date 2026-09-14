"""``generate()``: the "delivery" half of the v2 orchestrator, unwrapping
one ``Advice`` into rendered ``Step``s (v2 epic 8, #8). See
``_pipeline.py`` for the ``advise()``/``generate()`` promise this splits.
"""

from __future__ import annotations

from typing import Literal

from goldilocks_core.advisors.relax import RelaxOptions, VcRelaxOptions
from goldilocks_core.checks import CheckReport
from goldilocks_core.failures import ExpectedFailure
from goldilocks_core.generation.quantum_espresso.relax import write_qe_relax
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
    advice: Advice,
    report: CheckReport,
    *,
    ctx: SharedContext | None = None,
    purpose: Literal["scf", "nscf", "relax", "vc-relax"] = "scf",
    relax: RelaxOptions | VcRelaxOptions | None = None,
) -> tuple[Step, ...]:
    """Delivery: raises ``AdviceIncomplete`` if ``report`` is not ``ok``,
    per the "generate() refuses iff a needed field is unavailable/blocked"
    promise -- callers must call ``check(advice)`` first and are expected
    to inspect ``report.blocking`` themselves before ever reaching here
    in normal operation; this raise is the hard boundary, not the
    primary way a caller learns what is missing.

    ``purpose`` (v2 epic 9, #9): selects the writer and QE's own
    ``calculation`` value -- see ``write_qe_scf``/``write_qe_relax``'s
    own docstrings. Every existing caller renders an scf step;
    ``service/_dos.py``'s nscf pass is the first to pass
    ``purpose="nscf"``.

    ``relax`` (v2 epic 10, #10): only ``service/_relax.py`` ever passes
    this -- it is threaded onto ``PwSettings.relax`` and dispatches to
    ``write_qe_relax`` instead of ``write_qe_scf`` whenever ``purpose``
    is ``"relax"``/``"vc-relax"``. Reusing this function rather than a
    parallel ``generate_relax`` duplicating the ``SystemSettings``/
    ``PwSettings`` construction above: nothing about assembling those
    two differs between an scf step and a relax step, only which
    writer gets called and what ``PwSettings.relax`` holds.
    """
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
        relax=relax,
    )
    writer = write_qe_relax if purpose in ("relax", "vc-relax") else write_qe_scf
    return tuple(
        writer(
            system,
            step_settings,
            advice.step.resources.job.value,
            ctx or default_shared_context(),
            purpose,
        )
    )
