"""checks: the one place in the whole system allowed to refuse a request.

New in v2 (v2 epic 6, #6); no v1 precedent. Per
goldilocks-core-design.md's own summary (line 4007): "``check_all`` is
the sole gateway for cross-parameter constraints, and also the boundary
between ``advise`` and ``generate``." Every advisor in this codebase
degrades field-by-field (``Resolved``/``Unavailable``/``Blocked``)
instead of refusing a request, and only a fault (such as a worker process
that dies) raises; a runnable input is only refused here, by returning a
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
   *that shares parity with the electron count* (nelup=(nelec+tot)/2,
   neldw=(nelec-tot)/2, #70: both need to be whole numbers, which an
   integer ``tot_magnetization`` alone does not guarantee -- QE's own
   requirement once occupation numbers are no longer automatically
   split by smearing). This is scoped to the ``scf`` step only: an
   ``nscf`` step reads the charge/spin density a prior ``scf`` step
   already converged and does not re-derive this constraint
   (goldilocks-core-design.md:3417, "per-step constraint").
   ``advisors/occupations.py`` already flags this combination with an
   informational warning when it produces ``fixed``, but only *this*
   module owns the actual numeric check and the decision to block.

3. Its sibling rule, found empirically 2026-09-16 against real QE 7.3
   source (``PW/src/iweights.f90``'s ``iweights_only``): ``occupations
   ='fixed'`` together with a *non*-spin-polarized system requires an
   *even* ``electron_count`` -- with ``nspin=1`` (``degspin=2``), fixed
   occupations fill ``NINT(nelec)/degspin`` doubly-occupied bands, which
   can only reconstruct an even ``nelec`` exactly. An odd ``nelec`` here
   silently truncates to the nearest even count (QE only ever prints an
   ``infomsg`` -- "the system is metallic, specify occupations" --
   never an error), converging on a wrong total charge with nothing
   obviously wrong in the output. Same ``plan.SCF_LIKE_PURPOSES``
   scoping as rule 2, same reasoning.

4. A user ``ndiag`` whose s x s diagonalization grid cannot distribute
   ``nbnd`` bands (#133), s = isqrt(min(ndiag, ntasks/npool)): pw.x
   aborts at startup when s > nbnd (QE 7.3
   ``Modules/set_para_diag.f90:54-55``), and ScaLAPACK's block layout
   leaves a process column empty for subspace sizes up to (s-1)^2, which
   ELPA rejects and QE does not detect. The subspace never shrinks below
   nbnd, so nbnd > (s-1)^2 rules out both, for every purpose and on
   every profile.

**Everything else in the design doc's "constraint check table"** is
not a rule here. Memory overflow is a warning, not a block
(``job.memory_exceeds_node`` in ``advisors/job_resources.py``), like a
node count above the partition's ceiling: the memory figure is an
estimate. A disordered structure leaves the job unavailable, since no
written cell exists to size it for (``service/_step_resources.py``),
and that blocks through ``parallelisation``. The tri-tier
override-conflict rules are left as a gap. This module's job is to
establish *that* single gate exists and works for what is checkable
today, not to enumerate every rule up front.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from goldilocks_core.advisors.boundary import BoundaryFacts
from goldilocks_core.advisors.electron_count import ElectronCountDecision
from goldilocks_core.advisors.job_resources import JobDecision
from goldilocks_core.advisors.magnetic_config import MagneticConfigFacts
from goldilocks_core.advisors.nbnd import NbndDecision
from goldilocks_core.advisors.occupations import OccupationsDecision
from goldilocks_core.advisors.parallelisation import (
    ParallelisationDecision,
    diag_grid_side,
)
from goldilocks_core.advisors.relax import RelaxOptions, VcRelaxOptions
from goldilocks_core.advisors.vdw_method import VdwFacts
from goldilocks_core.analysis.geometry import (
    GeometryFacts,
    ab_in_xy_plane,
    vacuum_axes,
)
from goldilocks_core.functionals import D3BJ_FUNCTIONALS, has_d3bj_parameters
from goldilocks_core.plan import SCF_LIKE_PURPOSES
from goldilocks_core.resolution import Blocked, FieldState

_MISSING_TOT_MAGNETIZATION = (
    "occupations='fixed' with a spin-polarized system requires an explicit "
    "integer tot_magnetization, but none was set."
)
_EVEN_ELECTRON_COUNT_TOLERANCE = 1e-6
"""Matches QE's own ``eps8``-scale tolerance (``PW/src/setup.f90``'s odd
-electron check) for treating a heuristically-summed ``nelec`` as an
integer despite floating-point noise, without also treating a
genuinely fractional ``nelec`` (e.g. a disordered/partial-occupancy
structure) as spuriously even."""
_TWO_D_CELL_DOFREE = frozenset({"2Dxy", "2Dshape", "ibrav+2Dxy", "ibrav+2Dshape"})
"""``cell_dofree`` values that move only the x and y components of a and b.
The 'ibrav+' forms act as the plain ones under the writer's ``ibrav = 0``
(QE 7.3 Modules/latgen.f90:749-752)."""


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
    electron_count: FieldState[ElectronCountDecision] | None = None,
    relax: FieldState[RelaxOptions | VcRelaxOptions] | None = None,
    geometry: FieldState[GeometryFacts] | None = None,
    boundary: FieldState[BoundaryFacts] | None = None,
    vdw: FieldState[VdwFacts] | None = None,
    functional: FieldState[str] | None = None,
    job: FieldState[JobDecision] | None = None,
    parallel: FieldState[ParallelisationDecision] | None = None,
    nbnd: FieldState[NbndDecision] | None = None,
    purpose: str = "scf",
) -> CheckReport:
    """The single hard-fail gate.

    ``field_states`` is every ``FieldState`` a caller has assembled so
    far (from any advisor) -- any that are ``Blocked`` land in
    ``report.blocking`` via ``collect_blocked``. ``occupations``/
    ``magnetic``/``electron_count``/``relax``/``geometry``/``boundary``/
    ``vdw``/``functional``/``job``/``parallel``/``nbnd``/``purpose``
    additionally opt into the named cross-parameter rules this module
    implements; passing them here also folds them into the generic
    ``Blocked`` scan, so a caller does not need to repeat them in
    ``field_states`` as well.

    Each rule returns its blocking reason or ``None``; a new rule is one
    more entry in ``rule_reasons``.
    """
    optional = tuple(
        s
        for s in (
            occupations,
            magnetic,
            electron_count,
            relax,
            geometry,
            boundary,
            vdw,
            functional,
            job,
            parallel,
            nbnd,
        )
        if s is not None
    )
    rule_reasons = (
        _fixed_occupations_needs_integer_moment(
            occupations, magnetic, electron_count, purpose
        ),
        _fixed_occupations_needs_even_electron_count(
            occupations, magnetic, electron_count, purpose
        ),
        _vc_relax_requires_bfgs_ion_dynamics(relax, purpose),
        _vc_relax_forbids_isolated_boundary(boundary, purpose),
        _two_d_cell_dofree_needs_xy_frame(relax, magnetic),
        _dispersion_needs_parameterised_functional(vdw, functional),
        _fix_bottom_layers_requires_2d_geometry(relax, geometry),
        _npool_must_divide_ntasks(job, parallel),
        _ndiag_grid_must_distribute_bands(job, parallel, nbnd),
    )
    return CheckReport(
        blocking=(
            *collect_blocked(*field_states, *optional),
            *(reason for reason in rule_reasons if reason is not None),
        )
    )


def _fixed_occupations_needs_integer_moment(
    occupations: FieldState[OccupationsDecision] | None,
    magnetic: FieldState[MagneticConfigFacts] | None,
    electron_count: FieldState[ElectronCountDecision] | None,
    purpose: str,
) -> str | None:
    if purpose not in SCF_LIKE_PURPOSES or occupations is None or magnetic is None:
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
    if electron_count is not None and electron_count.ok:
        # nelup = (nelec + tot) / 2, neldw = (nelec - tot) / 2 (#70): both
        # need to be integers, which needs tot_magnetization and nelec to
        # share parity -- an integer tot_magnetization alone is not
        # sufficient (e.g. nelec=3, tot=2 gives nelup=2.5/neldw=0.5).
        nelup = (electron_count.value.nelec + tot) / 2
        if abs(round(nelup) - nelup) > _EVEN_ELECTRON_COUNT_TOLERANCE:
            return (
                "occupations='fixed' requires tot_magnetization to share "
                "parity with the electron count, so nelup/neldw are both "
                f"integers; got tot_magnetization={tot!r}, "
                f"nelec={electron_count.value.nelec!r}."
            )
    return None


def _fixed_occupations_needs_even_electron_count(
    occupations: FieldState[OccupationsDecision] | None,
    magnetic: FieldState[MagneticConfigFacts] | None,
    electron_count: FieldState[ElectronCountDecision] | None,
    purpose: str,
) -> str | None:
    """Sibling of ``_fixed_occupations_needs_integer_moment`` above, for
    the non-spin-polarized side: found empirically 2026-09-16, verified
    against real QE 7.3 source (``PW/src/iweights.f90``) -- see this
    module's own docstring, rule 3."""
    if (
        purpose not in SCF_LIKE_PURPOSES
        or occupations is None
        or magnetic is None
        or electron_count is None
    ):
        return None
    if not (occupations.ok and magnetic.ok and electron_count.ok):
        return None  # already surfaced via collect_blocked, or genuinely unknown
    if occupations.value.occupations != "fixed" or magnetic.value.spin_polarized:
        return None
    nelec = electron_count.value.nelec
    if abs(round(nelec / 2) - nelec / 2) > _EVEN_ELECTRON_COUNT_TOLERANCE:
        return (
            "occupations='fixed' with a non-spin-polarized system requires "
            f"an even electron count; got nelec={nelec!r}, which likely means "
            "an unpaired electron -- set spin_polarized=true, or "
            "occupations=smearing/tetrahedra_opt if this is genuinely metallic."
        )
    return None


def _npool_must_divide_ntasks(
    job: FieldState[JobDecision] | None,
    parallel: FieldState[ParallelisationDecision] | None,
) -> str | None:
    """``npool`` not dividing ``ntasks`` is a hard MPI-layout requirement
    QE itself refuses (``advisors/parallelisation.py``'s own docstring):
    a bad ``npool`` fails only once compute is already allocated and the
    job is running, worse than the same "warn, don't block" pattern used
    for e.g. ``nodes``/``walltime_h`` exceeding a partition's ceiling
    (which fail at SLURM *submission*, before any compute is consumed).
    ``parallelisation()``'s own heuristic default (``best_npool``)
    always returns a value that divides ``ntasks`` by construction, so
    this can only ever fire for a human override."""
    if job is None or parallel is None or not job.ok or not parallel.ok:
        return None
    if job.value.ntasks % parallel.value.npool != 0:
        return (
            f"npool={parallel.value.npool} does not divide "
            f"ntasks={job.value.ntasks}; QE refuses this MPI layout."
        )
    return None


def _ndiag_grid_must_distribute_bands(
    job: FieldState[JobDecision] | None,
    parallel: FieldState[ParallelisationDecision] | None,
    nbnd: FieldState[NbndDecision] | None,
) -> str | None:
    """Rule 4 of this module's docstring. ``parallelisation()`` never
    sets ``ndiag`` itself, so this only fires for a human override; like
    a bad ``npool``, such a layout fails only after queueing, or hangs a
    node until the walltime runs out."""
    if job is None or parallel is None or nbnd is None:
        return None
    if not (job.ok and parallel.ok and nbnd.ok) or parallel.value.ndiag is None:
        return None  # already surfaced via collect_blocked, or left to pw.x
    ndiag = parallel.value.ndiag
    bands = nbnd.value.nbnd
    side = diag_grid_side(ndiag, job.value.ntasks // parallel.value.npool, bands)
    if bands > (side - 1) ** 2:
        return None
    return (
        f"ndiag={ndiag} gives a {side}x{side} diagonalization grid, which "
        f"cannot distribute nbnd={bands} bands (it needs nbnd > "
        f"{(side - 1) ** 2}); pw.x would abort at startup or fail in the "
        "parallel eigensolver. Unset ndiag so pw.x chooses it from nbnd, "
        f"or use ndiag <= {(math.isqrt(bands - 1) + 1) ** 2}."
    )


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


def _vc_relax_forbids_isolated_boundary(
    boundary: FieldState[BoundaryFacts] | None, purpose: str
) -> str | None:
    """Under Martyna-Tuckerman, from any source, pw.x clears ``tstress``
    (QE 7.3 PW/src/input.f90:1361-1365) even though vc-relax had switched
    it on (input.f90:646): BFGS then sees a zero cell gradient and reports
    the input cell as converged, exiting 0 (#126)."""
    if purpose != "vc-relax" or boundary is None or not boundary.ok:
        return None
    if boundary.value.assume_isolated != "martyna-tuckerman":
        return None
    return (
        "vc-relax cannot relax the cell with assume_isolated='martyna-tuckerman': "
        "pw.x computes no stress for an isolated system, so the input cell "
        "would be reported as converged. Use task='relax' for a molecule or "
        "cluster in a box, or set assume_isolated='none' for a periodic structure."
    )


def _two_d_cell_dofree_needs_xy_frame(
    relax: FieldState[RelaxOptions | VcRelaxOptions] | None,
    magnetic: FieldState[MagneticConfigFacts] | None,
) -> str | None:
    """QE's 2D cell constraints move only the x and y components of a and b
    and hold c fixed, in Cartesian axes (QE 7.3 Modules/cell_base.f90:742-749
    and :803-808), and BFGS judges convergence on the masked stress. With a
    or b out of the xy plane they distort any cell, bulk included: SCARF
    shrank a and b of rhombohedral Bi2Se3 under '2Dxy' while c stayed fixed.
    With vacuum along a or b they free a vacuum vector and freeze one in the
    sheet (#132). Checked on the cell the writer renders, whatever the
    source of ``cell_dofree``."""
    if relax is None or magnetic is None or not (relax.ok and magnetic.ok):
        return None
    options = relax.value
    if not isinstance(options, VcRelaxOptions):
        return None
    if options.cell_dofree not in _TWO_D_CELL_DOFREE:
        return None
    cell = magnetic.value.relabeled_structure
    found = []
    if not ab_in_xy_plane(cell.lattice):
        found.append("a or b out of the xy plane")
    axes = vacuum_axes(cell)
    if axes not in ((), ("c",)):
        found.append(f"vacuum along {' and '.join(axes)}")
    if not found:
        return None
    return (
        f"cell_dofree={options.cell_dofree!r} moves only the x and y components of "
        "a and b and holds c fixed, so it needs a and b in the Cartesian xy plane "
        "and any vacuum along c, but the cell to be written has "
        f"{' and '.join(found)}. Set cell_dofree='all', or reorient the cell."
    )


def _dispersion_needs_parameterised_functional(
    vdw: FieldState[VdwFacts] | None, functional: FieldState[str] | None
) -> str | None:
    """D3(BJ), the only dispersion method generated, needs damping
    parameters for the functional: pw.x passes DFT-D3 the pseudopotentials'
    short functional name (QE 7.3 PW/src/input.f90:2039-2041) and stops with
    "functional name unknown" when the table lacks it, as for LDA (#141).
    ``vdw_method`` never picks D3 for such a functional itself, so this
    refuses ``use_vdw=True`` from any source."""
    if vdw is None or functional is None or not (vdw.ok and functional.ok):
        return None
    if not vdw.value.use_vdw or has_d3bj_parameters(functional.value):
        return None
    return (
        "use_vdw=true adds D3(BJ) dispersion, but Quantum ESPRESSO's DFT-D3 has "
        f"no parameters for functional {functional.value!r}, so pw.x would stop "
        "with 'functional name unknown'. Set use_vdw=false, or choose a "
        f"functional with D3(BJ) parameters ({', '.join(sorted(D3BJ_FUNCTIONALS))})."
    )


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
