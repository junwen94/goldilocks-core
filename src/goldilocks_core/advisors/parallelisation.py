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
(goldilocks-core-design.md:1396-1407).

**`ndiag` is left to pw.x unless the user sets it (#133).** Without
`-ndiag`, pw.x (QE >= 7.1) picks the largest square np^2 with
np <= nint(nbnd/100) that fits the processes per pool, so 1 below 150
bands (QE 7.3 `PW/src/setup.f90:827-832`). A user `ndiag` runs on an
s x s grid (`diag_grid_side`) and can stop pw.x: it aborts at startup
when s > nbnd (`Modules/set_para_diag.f90:54-55`), and ScaLAPACK's
block layout leaves a process column without rows of an n x n subspace
matrix for some n <= (s-1)^2 (`LAXlib/la_helper.f90:468-481`), which
ELPA rejects and QE does not detect (`LAXlib/zhpev_drv.f90:1513`).
`checks.py`'s `_ndiag_grid_must_distribute_bands` refuses such a value;
this module only carries it. Without ScaLAPACK, LAXlib still honours
`-ndiag` with its own distributed solver
(`Modules/set_para_diag.f90:79-83`).

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
        code="parallelisation.ndiag_not_a_square_integer",
        level="info",
        category="parallelisation",
        message=(
            "ndiag is not a square integer; pw.x runs the largest square "
            "grid that fits and never rejects the value."
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
    """``npool``/``ndiag`` must be positive (#35, v2 epic 9, #9): before
    this, ``npool=0`` reached ``job.ntasks % npool`` as a raw
    ``ZeroDivisionError`` and a negative ``ndiag`` reached ``math.isqrt``.
    The layout itself is refused in ``checks.py``, not here: an ``npool``
    that does not divide ``ntasks`` (``_npool_must_divide_ntasks``) and
    an ``ndiag`` whose grid cannot distribute ``nbnd`` bands
    (``_ndiag_grid_must_distribute_bands``, #133). A non-square ``ndiag``
    only gets an info note: pw.x runs the largest square grid that fits."""

    npool: int | None = Field(default=None, gt=0)
    ndiag: int | None = Field(default=None, gt=0)


def parallelisation(
    job: JobDecision,
    n_irr_k: int | None = None,
    human: ParallelisationHumanInput | None = None,
) -> FieldState[ParallelisationDecision]:
    """``npool`` from ``job.ntasks`` and ``n_irr_k`` unless ``human``
    sets it; ``ndiag`` only from ``human`` (``None`` leaves it to pw.x)."""
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

    if human.ndiag is not None:
        source = "human"
        warnings.extend(_ndiag_notes(human.ndiag, job.ntasks // npool))

    decision = ParallelisationDecision(
        npool=npool, ndiag=human.ndiag, warnings=tuple(warnings)
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


def diag_grid_side(ndiag: int, procs_per_pool: int) -> int:
    """Side s of the s x s linear-algebra grid pw.x builds for
    ``-ndiag``: isqrt(min(ndiag, processes per pool)), at least 1 (QE
    7.3 ``LAXlib/la_helper.f90:125`` and ``:171-176``, ``GRID2D_DIMS``
    'S' in ``LAXlib/distools.f90:84``). pw.x takes the processes of a
    band group; without ``-nband`` that is the whole pool."""
    return max(1, math.isqrt(min(ndiag, procs_per_pool)))


def _ndiag_notes(ndiag: int, procs_per_pool: int) -> list[Warning]:
    root = math.isqrt(ndiag)
    if root * root == ndiag:
        return []
    side = diag_grid_side(ndiag, procs_per_pool)
    return [
        Warning(
            code="parallelisation.ndiag_not_a_square_integer",
            level="info",
            category="parallelisation",
            message=(
                f"ndiag={ndiag} is not a square integer; pw.x runs a "
                f"{side}x{side} grid ({side * side} processes) and never "
                "rejects the value."
            ),
        )
    ]
