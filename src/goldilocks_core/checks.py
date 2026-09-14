"""checks: the one place in the whole system allowed to hard-fail.

New in v2 (v2 epic 6, #6); no v1 precedent. Per
goldilocks-core-design.md's own summary (line 4007): "``check_all`` is
the sole gateway for cross-parameter constraints, and also the boundary
between ``advise`` and ``generate``." Every advisor in this codebase
degrades field-by-field (``Resolved``/``Unavailable``/``Blocked``) and
never raises; a runnable input is only refused here, by returning a
non-empty ``CheckReport.blocking`` for a future ``generate()`` to read
-- not by raising an exception. **By convention, any other ``raise`` in
generation/advisor code found after this epic should be treated as
suspicious** (the issue's own acceptance criterion), since this module
is now where that decision belongs.

**Two independent mechanisms, not one:**

1. ``collect_blocked`` -- cross-parameter constraints are not the only
   way to end up unable to generate a runnable input: any upstream
   advisor that itself resolved to ``Blocked`` should also end up in
   ``report.blocking``, with its root cause surfaced (the design doc's
   own "blocking is derivable... the rejection reason follows the
   ``blocked_by`` chain back to the root cause, no separate
   error-message system needed", goldilocks-core-design.md:3261-3263).
   This mechanism is generic: it takes any ``FieldState`` from any
   advisor, not just the ones this epic happens to build.

2. A named cross-parameter rule -- ``occupations='fixed'`` together with
   a spin-polarized system requires an *integer* ``tot_magnetization``
   (QE's own requirement once occupation numbers are no longer
   automatically split by smearing). This is scoped to the ``scf``
   step only: an ``nscf`` step reads the charge/spin density a prior
   ``scf`` step already converged and does not re-derive this
   constraint (goldilocks-core-design.md:3417, "per-step constraint").
   ``advisors/occupations.py`` already flags this combination with an
   informational warning when it produces ``fixed``, but only *this*
   module owns the actual numeric check and the decision to block.

**Everything else in the design doc's "constraint check table"**
(memory overflow, disordered-structure generation blocking, the
tri-tier override-conflict rules) needs infrastructure this epic does
not build -- HPC resource estimates, a real orchestrator assembling
``system_settings``/``per_step`` -- and is deliberately left as a gap
for whichever future epic adds it, not attempted here. This module's
job is to establish *that* single gate exists and works for what is
checkable today, not to enumerate every rule up front.
"""

from __future__ import annotations

from dataclasses import dataclass

from goldilocks_core.advisors.job_resources import JobDecision
from goldilocks_core.advisors.magnetic_config import MagneticConfigFacts
from goldilocks_core.advisors.occupations import OccupationsDecision
from goldilocks_core.advisors.parallelisation import ParallelisationDecision
from goldilocks_core.advisors.relax import RelaxOptions, VcRelaxOptions
from goldilocks_core.analysis.geometry import GeometryFacts
from goldilocks_core.resolution import Blocked, FieldState

_MISSING_TOT_MAGNETIZATION = (
    "occupations='fixed' with a spin-polarized system requires an explicit "
    "integer tot_magnetization, but none was set."
)
_SCF_LIKE_PURPOSES = frozenset({"scf", "relax", "vc-relax"})
"""``purpose``s that run their own electronic-structure scf loop and so
still need the fixed-occupations/integer-moment rule below -- unlike
``nscf``, which reads a prior step's already-converged density/spin and
does not re-derive it (v2 epic 9, #9's original scoping). ``relax``/
``vc-relax`` re-run scf at every ionic step (v2 epic 10, #10), so the
same constraint applies to them too."""


@dataclass(frozen=True, slots=True)
class CheckReport:
    blocking: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.blocking


def collect_blocked(*field_states: FieldState[object]) -> tuple[str, ...]:
    """Every ``Blocked`` field state's root cause, in the order given.

    ``Unavailable`` and ``Resolved`` states are not blocking on their
    own -- an advisor that could not tell, or a caller that never
    supplied a given field at all, is not the same as an upstream
    failure. Only genuine ``Blocked`` propagation reaches here.
    """
    return tuple(
        state.root_cause() for state in field_states if isinstance(state, Blocked)
    )


def check_all(
    *field_states: FieldState[object],
    occupations: FieldState[OccupationsDecision] | None = None,
    magnetic: FieldState[MagneticConfigFacts] | None = None,
    relax: FieldState[RelaxOptions | VcRelaxOptions] | None = None,
    geometry: FieldState[GeometryFacts] | None = None,
    job: FieldState[JobDecision] | None = None,
    parallel: FieldState[ParallelisationDecision] | None = None,
    purpose: str = "scf",
) -> CheckReport:
    """The single hard-fail gate.

    ``field_states`` is every ``FieldState`` a caller has assembled so
    far (from any advisor) -- any that are ``Blocked`` land in
    ``report.blocking`` via ``collect_blocked``. ``occupations``/
    ``magnetic``/``relax``/``geometry``/``job``/``parallel``/``purpose``
    additionally opt into the named cross-parameter rules this module
    implements; passing them here also folds them into the generic
    ``Blocked`` scan, so a caller does not need to repeat them in
    ``field_states`` as well.
    """
    optional = tuple(
        s
        for s in (occupations, magnetic, relax, geometry, job, parallel)
        if s is not None
    )
    blocking = list(collect_blocked(*field_states, *optional))

    reason = _fixed_occupations_needs_integer_moment(occupations, magnetic, purpose)
    if reason is not None:
        blocking.append(reason)

    reason = _vc_relax_requires_bfgs_ion_dynamics(relax, purpose)
    if reason is not None:
        blocking.append(reason)

    reason = _fix_bottom_layers_requires_2d_geometry(relax, geometry)
    if reason is not None:
        blocking.append(reason)

    reason = _npool_must_divide_ntasks(job, parallel)
    if reason is not None:
        blocking.append(reason)

    return CheckReport(blocking=tuple(blocking))


def _fixed_occupations_needs_integer_moment(
    occupations: FieldState[OccupationsDecision] | None,
    magnetic: FieldState[MagneticConfigFacts] | None,
    purpose: str,
) -> str | None:
    if purpose not in _SCF_LIKE_PURPOSES or occupations is None or magnetic is None:
        return None
    if not (occupations.ok and magnetic.ok):
        return None  # already surfaced via collect_blocked, or genuinely unknown
    if occupations.value.occupations != "fixed" or not magnetic.value.spin_polarized:
        return None
    tot = magnetic.value.tot_magnetization
    if tot is None:
        return _MISSING_TOT_MAGNETIZATION
    if tot != int(tot):
        return (
            "occupations='fixed' with a spin-polarized system requires an "
            f"integer tot_magnetization; got {tot!r}."
        )
    return None


def _npool_must_divide_ntasks(
    job: FieldState[JobDecision] | None,
    parallel: FieldState[ParallelisationDecision] | None,
) -> str | None:
    """``npool`` not dividing ``ntasks`` is a hard MPI-layout requirement
    QE itself refuses (``advisors/parallelisation.py``'s own docstring) --
    unlike ``ndiag``'s squareness/``has_scalapack`` gate (a genuine
    QE-will-likely-ignore-or-reject *preference*, correctly left
    advisory), a bad ``npool`` here fails only once compute is already
    allocated and the job is running, worse than the same "warn, don't
    block" pattern used for e.g. ``nodes``/``walltime_h`` exceeding a
    partition's ceiling (which fail at SLURM *submission*, before any
    compute is consumed). ``parallelisation()``'s own heuristic default
    (``_best_npool``) always returns a value that divides ``ntasks`` by
    construction, so this can only ever fire for a human override."""
    if job is None or parallel is None or not job.ok or not parallel.ok:
        return None
    if job.value.ntasks % parallel.value.npool != 0:
        return (
            f"npool={parallel.value.npool} does not divide "
            f"ntasks={job.value.ntasks}; QE refuses this MPI layout."
        )
    return None


def _vc_relax_requires_bfgs_ion_dynamics(
    relax: FieldState[RelaxOptions | VcRelaxOptions] | None,
    purpose: str,
) -> str | None:
    """QE couples ``cell_dynamics`` to ``ion_dynamics`` for vc-relax
    (``advisors/relax.py``'s own docstring, audit finding B5): this
    codebase's ``VcRelaxOptions.cell_dynamics`` is a derived property
    that always mirrors ``ion_dynamics``, which only produces a valid QE
    keyword when ``ion_dynamics == 'bfgs'`` (the only vc-relax
    ``cell_dynamics`` value this codebase's generation layer renders).
    ``ion_dynamics`` in {'damp', 'fire'} is real, QE-documented syntax
    for a plain ``relax`` -- only vc-relax is restricted here."""
    if purpose != "vc-relax" or relax is None or not relax.ok:
        return None
    if relax.value.ion_dynamics != "bfgs":
        return (
            "vc-relax requires ion_dynamics='bfgs' in this codebase (got "
            f"{relax.value.ion_dynamics!r}): QE couples cell_dynamics to "
            "ion_dynamics for vc-relax, and only the bfgs/bfgs combination "
            "is modelled here -- damp-paired cell dynamics is a future "
            "epic's scope."
        )
    return None


def _fix_bottom_layers_requires_2d_geometry(
    relax: FieldState[RelaxOptions | VcRelaxOptions] | None,
    geometry: FieldState[GeometryFacts] | None,
) -> str | None:
    """``relax.fix_bottom_layers`` (#44) fixes atoms in a slab's bottom
    N atomic layers -- meaningless (there is no "bottom" along a
    stacking direction) unless the structure actually is a 2D slab.
    Blocked here rather than silently ignored or left to fail deep
    inside generation's own layer-detection algorithm with a less
    actionable message."""
    if relax is None or not relax.ok or relax.value.fix_bottom_layers is None:
        return None
    if geometry is None or not geometry.ok:
        return (
            "relax.fix_bottom_layers requires a 2D slab structure, but "
            "geometry classification is not available."
        )
    if geometry.value.dimensionality != "2d":
        return (
            "relax.fix_bottom_layers requires a 2D slab structure; got "
            f"dimensionality={geometry.value.dimensionality!r}."
        )
    return None
