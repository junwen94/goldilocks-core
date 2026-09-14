"""parallelisation: npool/ndiag, from a job's ntasks and the step's
n_irr_k -- the other half of the false-circularity fix
`advisors/job_resources.py` describes: this reads `job.ntasks`, never
the reverse.

New in v2 (v2 epic 7, #7); no v1 precedent.

**Full constraint set** (goldilocks-core-design.md:1487-1493):
`npool` must divide `ntasks` (a hard MPI-layout requirement -- QE
itself refuses an `ntasks`/`npool` combination that does not divide
evenly); `npool` should also divide `n_reduced_kpoints`, or some pools
sit idle once every k-point has been assigned to a busy pool
(goldilocks-core-design.md:1490, ":1511-1516" -- this is exactly why
epic 6's `kmesh.py`/`advisors/n_irr_k.py` port matters here: v1 never
had a real `n_reduced_kpoints`, so v1's own parallel advice was always
a guess). This module only enforces the hard constraint and *prefers*
the soft one; it does not search for a global optimum -- picking the
actually-fastest layout is explicitly named as needing ml eventually
(goldilocks-core-design.md:1396-1407: `npool`/`ndiag` are an efficiency
question, not a correctness one, with much smaller error-cost asymmetry
than `walltime`).

**`ndiag`, gated on the HPC profile's `has_scalapack`**
(goldilocks-core-design.md's `[codes.quantum_espresso] has_scalapack`
profile field, matching the official user guide's own default-selection
rule, verified directly 2026-09-14 against
quantum-espresso.org/Doc/user_guide/node20.html -- not just the design
doc's earlier citation of it): "nd is set to 1 if ScaLAPACK is not
compiled, it is set to the square integer smaller than or equal to the
number of processors of each pool." That page separately states the
linear-algebra group size must be "smaller than" (strict) the pool's
processor count as a general constraint -- a QE-documentation
inconsistency in its own right (the two statements can disagree exactly
when a pool's size is itself a perfect square), not a misreading here:
this module replicates QE's own stated *default-selection* procedure
verbatim (`<=`), which is what a caller actually wants reproduced.
`-nband`/`-nb` (band groups, useful for hybrid functionals) and
`-ntg`/`-nt` (task groups, FFT parallelization for very large process
counts) are real, separate QE parallelization layers this module does
not model at all -- no advisor in this codebase needs them yet
(hybrid-functional support, and job sizes large enough for FFT-plane
-count to bind, are both out of scope today).

**`nimage` is always `None` here.** It only applies to NEB/phonon tasks
(`neb.x`'s images, `ph.x`'s irreps/q-points), and this codebase has no
NEB/phonon per-step advisor yet to decide how many images/q-points
there even are -- a documented gap, not a default of 1 masquerading as
a decision.

No `llm` override: `npool`/`ndiag` are technical MPI-layout tuning
knobs, not a scientific judgement call -- the same category
`advisors/n_irr_k.py`'s `nosym` put itself in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pydantic import Field

from goldilocks_core.advisors.job_resources import JobDecision
from goldilocks_core.inputs.overrides import HumanInput
from goldilocks_core.resolution import FieldState, Provenance, Resolved, Warning

WARNING_CATALOGUE = (
    Warning(
        code="parallelisation.npool_does_not_divide_ntasks",
        level="warning",
        category="parallelisation",
        message="npool does not divide ntasks; QE will refuse this layout.",
    ),
    Warning(
        code="parallelisation.npool_exceeds_kpoints",
        level="warning",
        category="parallelisation",
        message=(
            "npool exceeds the number of reduced k-points; at least one pool "
            "will have no k-point to work on."
        ),
    ),
    Warning(
        code="parallelisation.npool_load_imbalance",
        level="info",
        category="parallelisation",
        message=(
            "npool divides ntasks but not the number of reduced k-points "
            "evenly; some pools will handle one more k-point than others."
        ),
    ),
    Warning(
        code="parallelisation.ndiag_ignored_no_scalapack",
        level="warning",
        category="parallelisation",
        message=(
            "ndiag was given but the code was not built with ScaLAPACK on "
            "this profile; QE ignores -ndiag in that case."
        ),
    ),
    Warning(
        code="parallelisation.ndiag_not_a_square_integer",
        level="warning",
        category="parallelisation",
        message=(
            "ndiag is not a square integer; QE's own default-selection rule "
            "always picks one, and a non-square value may be rejected."
        ),
    ),
)
"""Every warning code this module can emit -- ``capabilities.py``'s
``warnings[]`` catalogue aggregates one of these tuples per advisor. The
real, per-occurrence messages (naming the actual npool/ntasks/k-point
figures) are built at their call sites below; these are the generic
descriptions of each code, for a caller that has not seen it fire."""


@dataclass(frozen=True, slots=True)
class ParallelisationDecision:
    npool: int
    ndiag: int | None
    nimage: int | None = None
    warnings: tuple[Warning, ...] = ()


class ParallelisationHumanInput(HumanInput):
    """``npool``/``ndiag`` must be positive (#35, v2 epic 9, #9) --
    before this, ``npool=0`` reached ``job.ntasks % npool`` as a raw
    ``ZeroDivisionError`` and a negative ``npool``/``ndiag`` reached
    ``math.isqrt`` on a negative operand, both unhandled Python
    exceptions instead of a clean ``InvalidSetting`` error. Squareness
    and the ``has_scalapack`` gate on ``ndiag`` stay advisory warnings
    (``_ndiag_advisory_warnings`` below) -- genuine QE-will-likely-ignore
    -or-reject *preferences*, not the same category as ``npool`` not
    dividing ``ntasks``, which is a hard MPI-layout requirement QE
    itself refuses outright (found not to be enforced during the epic
    5/6/7 delivery-layer audit; now a blocking rule in ``checks.py``'s
    ``_npool_must_divide_ntasks``, not just the warning still emitted
    below)."""

    npool: int | None = Field(default=None, gt=0)
    ndiag: int | None = Field(default=None, gt=0)


def parallelisation(
    job: JobDecision,
    has_scalapack: bool,
    n_irr_k: int | None = None,
    human: ParallelisationHumanInput | None = None,
) -> FieldState[ParallelisationDecision]:
    human = human or ParallelisationHumanInput()

    if human.npool is not None:
        npool = human.npool
        warnings: list[Warning] = []
        if job.ntasks % npool != 0:
            warnings.append(
                Warning(
                    code="parallelisation.npool_does_not_divide_ntasks",
                    level="warning",
                    category="parallelisation",
                    message=(
                        f"npool={npool} does not divide ntasks={job.ntasks}; QE "
                        "will refuse this layout."
                    ),
                )
            )
        if n_irr_k is not None and npool > n_irr_k:
            warnings.append(
                Warning(
                    code="parallelisation.npool_exceeds_kpoints",
                    level="warning",
                    category="parallelisation",
                    message=(
                        f"npool={npool} exceeds n_reduced_kpoints={n_irr_k}; at "
                        f"least {npool - n_irr_k} pool(s) will have no k-point "
                        "to work on."
                    ),
                )
            )
        source = "human"
    else:
        npool, warnings = _best_npool(job.ntasks, n_irr_k)
        source = "heuristic"

    ndiag = (
        human.ndiag
        if human.ndiag is not None
        else _ndiag(job.ntasks, npool, has_scalapack)
    )
    if human.ndiag is not None:
        source = "human"
        warnings.extend(_ndiag_advisory_warnings(human.ndiag, has_scalapack))

    decision = ParallelisationDecision(
        npool=npool, ndiag=ndiag, warnings=tuple(warnings)
    )
    return Resolved(decision, Provenance(source=source))


def _best_npool(ntasks: int, n_irr_k: int | None) -> tuple[int, list[Warning]]:
    """The largest divisor of ``ntasks`` that does not exceed ``n_irr_k`` --
    capping at ``n_irr_k`` is what prevents an idle pool (one with zero
    k-points to work on) outright; preferring the largest such divisor
    maximises k-point parallelism. Only when that divisor does not also
    divide ``n_irr_k`` evenly is there a warning -- that is a milder,
    unavoidable load *imbalance* (some pools get one more k-point than
    others), not idling, and is not worth sacrificing parallelism for by
    falling back to a smaller, evenly-dividing npool such as 1.
    """
    if n_irr_k is None:
        return 1, []

    candidates = [d for d in range(1, ntasks + 1) if ntasks % d == 0 and d <= n_irr_k]
    chosen = max(candidates)  # 1 always qualifies, so candidates is never empty
    if n_irr_k % chosen == 0:
        return chosen, []
    return chosen, [
        Warning(
            code="parallelisation.npool_load_imbalance",
            level="info",
            category="parallelisation",
            message=(
                f"npool={chosen} divides ntasks but not n_reduced_kpoints "
                f"({n_irr_k}) evenly; some pools will handle one more "
                "k-point than others."
            ),
        )
    ]


def _ndiag(ntasks: int, npool: int, has_scalapack: bool) -> int | None:
    if not has_scalapack:
        return None
    per_pool = ntasks // npool
    root = math.isqrt(per_pool)
    return max(1, root * root)


def _ndiag_advisory_warnings(ndiag: int, has_scalapack: bool) -> list[Warning]:
    """Advisory, not blocking (#35, v2 epic 9, #9) -- a human-supplied
    ``ndiag`` that is not square, or that is given when the profile has
    no ScaLAPACK, is a QE-will-likely-ignore-or-reject situation, the
    same category ``npool_does_not_divide_ntasks`` above already treats
    as warn-don't-block."""
    warnings: list[Warning] = []
    if not has_scalapack:
        warnings.append(
            Warning(
                code="parallelisation.ndiag_ignored_no_scalapack",
                level="warning",
                category="parallelisation",
                message=(
                    f"ndiag={ndiag} was given but this profile's code was not "
                    "built with ScaLAPACK; QE ignores -ndiag in that case."
                ),
            )
        )
    root = math.isqrt(ndiag)
    if root * root != ndiag:
        warnings.append(
            Warning(
                code="parallelisation.ndiag_not_a_square_integer",
                level="warning",
                category="parallelisation",
                message=(
                    f"ndiag={ndiag} is not a square integer; QE's own "
                    "default-selection rule always picks one, and a "
                    "non-square value may be rejected."
                ),
            )
        )
    return warnings
