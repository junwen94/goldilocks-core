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
   a collinear spin-polarized run (``MagneticConfigFacts.spin_mode``
   ``collinear``, nspin=2) requires an *integer* ``tot_magnetization``
   *that shares parity with the electron count* (nelup=(nelec+tot)/2,
   neldw=(nelec-tot)/2, #70: both need to be whole numbers, which an
   integer ``tot_magnetization`` alone does not guarantee -- QE's own
   requirement once occupation numbers are no longer automatically
   split by smearing). pw.x stops on fixed occupations without one only
   with lsda (QE 7.3 ``PW/src/input.f90:764-781``); a noncollinear
   (spin-orbit) run has lsda off and takes no ``tot_magnetization`` at
   all (``input.f90:761-762``), so it needs none (#146). This is scoped
   to the ``scf`` step only: an ``nscf`` step reads the charge/spin
   density a prior ``scf`` step already converged and does not re-derive
   this constraint (goldilocks-core-design.md:3417, "per-step
   constraint").
   The moment checked is the occupations decision's
   (``OccupationsDecision.tot_magnetization``), the one the writer
   renders (#138). ``advisors/occupations.py`` picks fixed for a
   collinear run only with a moment that passes, so this rule fires only
   for an explicit ``occupations='fixed'``, which that advisor flags with
   a warning; only *this* module blocks.

3. Its sibling rule, found empirically 2026-09-16 against real QE 7.3
   source (``PW/src/iweights.f90``'s ``iweights_only``): ``occupations
   ='fixed'`` together with a *non*-spin-polarized run (spin mode
   ``none``, or a spin-orbit run without moments) requires an *even*
   ``electron_count`` -- with ``nspin=1``
   (``degspin=2``), fixed occupations fill ``NINT(nelec)/degspin``
   doubly-occupied bands, which can only reconstruct an even ``nelec``
   exactly. An odd ``nelec`` here silently truncates to the nearest even
   count (QE only ever prints an ``infomsg`` -- "the system is metallic,
   specify occupations" -- never an error), converging on a wrong total
   charge with nothing obviously wrong in the output. A noncollinear run
   fills single-electron spinor bands (``degspin=1``,
   ``iweights.f90:105-107``), so its count is never truncated, but one
   without moments keeps time reversal (``domag`` false), so every band
   has a Kramers partner degenerate with it at the time-reversal-invariant
   k-points, and at every k with inversion: an odd count half-fills a
   degenerate pair, a metallic state fixed occupations cannot describe,
   and the rule applies to it too. Only a spin-polarized noncollinear run,
   which breaks time reversal, is exempt (#146). Same
   ``plan.SCF_LIKE_PURPOSES`` scoping as
   rule 2, same reasoning; like rule 2 it is reachable only from an
   explicit ``occupations='fixed'``, since the advisor's default for an odd
   count is smearing (#138).

4. ``nbnd`` must hold every electron of each spin channel (#131): pw.x
   stops in setup below ``NINT(nelec/2)`` ("too few bands"), below
   ``NINT(nelup)`` or ``NINT(neldw)`` ("too few spin up/dw bands") and, for
   a noncollinear run, below ``NINT(nelec)`` (QE 7.3
   ``PW/src/setup.f90:349-356``), with nelup/neldw from the occupations
   decision's ``tot_magnetization`` (``advisors/nbnd.py``'s
   ``band_floors``). The default never falls below these, so this refuses
   a human ``nbnd`` or a human or llm ``extra_bands``: ``nbnd=2`` on 8
   electrons, or 96 bands for LiFePO4 with ``tot_magnetization=16``, whose
   spin-up channel holds 104 electrons, used to pass and stop pw.x after
   the job had queued. pw.x checks only scf-like runs, but an nscf step
   needs every occupied band too, to place the Fermi level, so this
   applies to every purpose.

5. A user ``ndiag`` whose s x s diagonalization grid cannot distribute
   ``nbnd`` bands (#133), s = isqrt(min(ndiag, ntasks/npool)): pw.x
   aborts at startup when s > nbnd (QE 7.3
   ``Modules/set_para_diag.f90:54-55``), and ScaLAPACK's block layout
   leaves a process column empty for subspace sizes up to (s-1)^2, which
   ELPA rejects and QE does not detect. The subspace never shrinks below
   nbnd, so nbnd > (s-1)^2 rules out both, for every purpose and on
   every profile.

6. A vc-relax ``cell_dofree`` that masks Cartesian components of the
   lattice vectors ('a', 'b', 'c', 'x', 'y', 'z', 'xy', 'xz', 'yz', 'xyz':
   QE 7.3 ``Modules/cell_base.f90:710-830``'s
   ``iforceh(i, j)`` is Cartesian component i of lattice vector j) on a
   cell the magnetic step wrote rotated (``MagneticConfigFacts.frame``,
   ``pw_symmetry.pw_frame``, #140): its x, y and z are no longer the
   uploaded cell's, so the mask would hold other components than the ones
   asked for. Masks of whole lattice vectors ('fixa', 'fixb', 'fixc', the
   'epitaxial' forms), 'all' and 'shape' keep their meaning; '2Dxy' and
   '2Dshape' are left to the 2D-frame rule, since they act on a and b in
   the plane they span once that plane is xy, however it got there.

**Findings, not prose (#151).** Every refusal is a ``Finding``: a code
from ``FINDING_CATALOGUE``, the setting keys it concerns (``fields``, as
``capabilities()`` names them), a user-facing ``message`` and its
``remedies``, each a complete override patch that resolves it (a ``None``
value clears that override). A catalogue entry fixes a code's fields and
remedies; each rule fills in the message of its occurrence, and drops a
remedy that would not hold for it (cell_dofree=all for a cell with vacuum,
which it would relax too). A value only the user can choose (a
``tot_magnetization``) is named in the message, never guessed in a remedy.
A ``Blocked`` chain reaches the gate as ``advice.blocked`` with its root
cause and no remedy: a refusal that needs one must be a named rule here.
``CheckReport`` keeps one finding per root cause, however many fields share
it.

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
from dataclasses import dataclass, replace

from pymatgen.core import Structure

from goldilocks_core.advisors.boundary import BoundaryFacts
from goldilocks_core.advisors.electron_count import ElectronCountDecision
from goldilocks_core.advisors.job_resources import JobDecision
from goldilocks_core.advisors.magnetic_config import MagneticConfigFacts
from goldilocks_core.advisors.nbnd import NbndDecision, band_floors
from goldilocks_core.advisors.occupations import (
    OccupationsDecision,
    whole_spin_channels,
)
from goldilocks_core.advisors.parallelisation import (
    ParallelisationDecision,
    diag_grid_side,
)
from goldilocks_core.advisors.relax import RelaxOptions, VcRelaxOptions, wire_period
from goldilocks_core.advisors.vdw_method import VdwFacts
from goldilocks_core.analysis.geometry import (
    LatticeAxis,
    ab_in_xy_plane,
    slab_layers,
    two_d_truncation_obstacles,
    vacuum_axes,
)
from goldilocks_core.functionals import D3BJ_FUNCTIONALS, has_d3bj_parameters
from goldilocks_core.plan import SCF_LIKE_PURPOSES
from goldilocks_core.resolution import Blocked, FieldState

_TWO_D_CELL_DOFREE = frozenset({"2Dxy", "2Dshape"})
"""``cell_dofree`` values that move only the x and y components of a and b."""
_CARTESIAN_CELL_DOFREE = frozenset(
    {"a", "b", "c", "x", "y", "z", "xy", "xz", "yz", "xyz"}
)
"""``cell_dofree`` values that hold or free single Cartesian components of
the lattice vectors (rule 6 of the module docstring)."""


@dataclass(frozen=True, slots=True)
class Finding:
    """One reason the gate refuses a request (module docstring). ``code`` is
    a stable ``FINDING_CATALOGUE`` key a caller may match on; ``fields`` are
    the ``capabilities()`` setting keys it concerns; ``message`` is the text
    a user reads, with the occurrence's values filled in; each of
    ``remedies`` is a complete override patch that resolves it, a ``None``
    value clearing that override."""

    code: str
    message: str
    fields: tuple[str, ...] = ()
    remedies: tuple[dict[str, object], ...] = ()

    def to_document(self) -> dict[str, object]:
        """The JSON every transport returns for this finding."""
        return {
            "code": self.code,
            "fields": list(self.fields),
            "message": self.message,
            "remedies": [dict(remedy) for remedy in self.remedies],
        }


_SMEARING = {"occupations": "smearing"}
_PERIODIC = {"isolation": "periodic"}
_MOMENT_FIELDS = ("occupations", "tot_magnetization")

_BLOCKED = Finding(
    code="advice.blocked",
    message=(
        "A decision the input needs could not be made; the message gives the "
        "root cause."
    ),
)
_MISSING_TOT_MAGNETIZATION = Finding(
    code="occupations.fixed_needs_tot_magnetization",
    message=(
        "occupations=fixed on a collinear spin-polarized run needs an integer "
        "tot_magnetization of the electron count's parity, and none was set or "
        "derived. Set occupations=smearing, or set tot_magnetization."
    ),
    fields=_MOMENT_FIELDS,
    remedies=(_SMEARING,),
)
_NON_INTEGER_TOT_MAGNETIZATION = Finding(
    code="occupations.fixed_needs_integer_tot_magnetization",
    message=(
        "occupations=fixed on a collinear spin-polarized run needs an integer "
        "tot_magnetization."
    ),
    fields=_MOMENT_FIELDS,
    remedies=(_SMEARING,),
)
_TOT_MAGNETIZATION_PARITY = Finding(
    code="occupations.fixed_tot_magnetization_parity",
    message=(
        "occupations=fixed needs tot_magnetization to share parity with the "
        "electron count."
    ),
    fields=_MOMENT_FIELDS,
    remedies=(_SMEARING,),
)
_ODD_ELECTRON_COUNT = Finding(
    code="occupations.fixed_needs_even_electron_count",
    message=(
        "occupations=fixed without spin polarization needs an even electron count."
    ),
    fields=("occupations", "spin_polarized"),
    remedies=(_SMEARING, {"spin_polarized": True, **_SMEARING}),
)
_TOO_FEW_BANDS = Finding(
    code="nbnd.too_few_bands",
    message=(
        "nbnd is below the band count pw.x needs for the electrons of each spin "
        "channel."
    ),
    fields=("nbnd", "extra_bands"),
    remedies=({"nbnd": None, "extra_bands": None},),
)
_NPOOL = Finding(
    code="parallelisation.npool_does_not_divide_ntasks",
    message="npool must divide ntasks.",
    fields=("npool", "ntasks"),
    remedies=({"npool": None},),
)
_NDIAG = Finding(
    code="parallelisation.ndiag_grid_too_large",
    message="ndiag gives a diagonalization grid too large for nbnd.",
    fields=("ndiag", "nbnd"),
    remedies=({"ndiag": None},),
)
_VC_RELAX_ION_DYNAMICS = Finding(
    code="relax.vc_relax_needs_bfgs",
    message="vc-relax is generated only with ion_dynamics=bfgs.",
    fields=("ion_dynamics",),
    remedies=({"ion_dynamics": "bfgs"},),
)
_ISOLATED_WITHOUT_VACUUM = Finding(
    code="boundary.isolated_needs_vacuum",
    message=(
        "isolation=isolated needs vacuum along a, b and c of the cell to be written."
    ),
    fields=("isolation",),
    remedies=(_PERIODIC,),
)
_SLAB_UNSUITED = Finding(
    code="boundary.slab_unsuited_to_2d_truncation",
    message=(
        "isolation=slab needs a cell that suits Quantum ESPRESSO's 2D Coulomb "
        "truncation."
    ),
    fields=("isolation",),
    remedies=(_PERIODIC,),
)
_ISOLATED_VC_RELAX = Finding(
    code="boundary.isolated_forbids_vc_relax",
    message=(
        "vc-relax cannot relax the cell with isolation=isolated "
        "(Martyna-Tuckerman): pw.x computes no stress for an isolated system, so "
        "the input cell would be reported as converged. Use the relax task for a "
        "molecule or cluster in a box, or set isolation=periodic for a periodic "
        "structure."
    ),
    fields=("isolation",),
)
"""No remedy: only a cell with vacuum along a, b and c may be isolated (the
vacuum rule refuses any other, offering isolation=periodic), and for such a
molecule or cluster the fix is the relax task, which no override expresses.
isolation=periodic would pass, but vc-relax would then move the vacuum
box."""
_SLAB_VC_RELAX_CELL_DOFREE = Finding(
    code="relax.slab_vc_relax_needs_2d_cell_dofree",
    message="vc-relax with isolation=slab needs a 2D cell_dofree.",
    fields=("cell_dofree", "isolation"),
    remedies=({"cell_dofree": "2Dxy"}, {**_PERIODIC, "cell_dofree": None}),
)
"""The periodic remedy also clears cell_dofree, so that the slab's default
'2Dxy' holds c, which is the vacuum, fixed: an explicit 'all' kept under
isolation=periodic would pass, but relax the vacuum. Both remedies hold only
on a slab with its vacuum along c in the xy frame, so the rule leaves any
other cell to the boundary rule's isolation=periodic."""
_TWO_D_CELL_DOFREE_FRAME = Finding(
    code="relax.two_d_cell_dofree_needs_xy_frame",
    message="A 2D cell_dofree needs a and b in the Cartesian xy plane.",
    fields=("cell_dofree",),
    remedies=({"cell_dofree": "all"},),
)
"""The remedy holds for a cell without vacuum only: with vacuum, 'all'
would relax the vacuum too, so the rule offers none there and names what
holds the vacuum fixed instead."""
_CARTESIAN_CELL_DOFREE_ROTATED = Finding(
    code="relax.cartesian_cell_dofree_on_rotated_cell",
    message=(
        "A cell_dofree that masks Cartesian components needs the cell written "
        "unrotated."
    ),
    fields=("cell_dofree",),
    remedies=({"cell_dofree": None},),
)
_DISPERSION_FUNCTIONAL = Finding(
    code="vdw.use_vdw_without_d3bj_parameters",
    message="use_vdw=true needs a functional with D3(BJ) parameters.",
    fields=("use_vdw", "functional"),
    remedies=({"use_vdw": False},),
)
_FIX_BOTTOM_LAYERS_WITHOUT_SLAB = Finding(
    code="relax.fix_bottom_layers_needs_slab",
    message="fix_bottom_layers needs a slab.",
    fields=("fix_bottom_layers",),
    remedies=({"fix_bottom_layers": None},),
)
_FIX_BOTTOM_LAYERS_EXCEEDS = Finding(
    code="relax.fix_bottom_layers_exceeds_layers",
    message="fix_bottom_layers exceeds the atomic layers of the slab.",
    fields=("fix_bottom_layers",),
    remedies=({"fix_bottom_layers": None},),
)

FINDING_CATALOGUE = (
    _BLOCKED,
    _MISSING_TOT_MAGNETIZATION,
    _NON_INTEGER_TOT_MAGNETIZATION,
    _TOT_MAGNETIZATION_PARITY,
    _ODD_ELECTRON_COUNT,
    _TOO_FEW_BANDS,
    _VC_RELAX_ION_DYNAMICS,
    _ISOLATED_WITHOUT_VACUUM,
    _SLAB_UNSUITED,
    _ISOLATED_VC_RELAX,
    _SLAB_VC_RELAX_CELL_DOFREE,
    _TWO_D_CELL_DOFREE_FRAME,
    _CARTESIAN_CELL_DOFREE_ROTATED,
    _DISPERSION_FUNCTIONAL,
    _FIX_BOTTOM_LAYERS_WITHOUT_SLAB,
    _FIX_BOTTOM_LAYERS_EXCEEDS,
    _NPOOL,
    _NDIAG,
)
"""Every finding the gate can return, with its code's fields and remedies
and a generic message; a rule returns its entry with the occurrence's
message, less any remedy that would not hold for it (module docstring)."""


@dataclass(frozen=True, slots=True)
class CheckReport:
    """The gate's verdict: ``blocking`` holds one finding per root cause, in
    first-seen order; findings with the same code and message are kept once,
    since many blocked fields can share one upstream cause."""

    blocking: tuple[Finding, ...] = ()

    def __post_init__(self) -> None:
        unique: dict[tuple[str, str], Finding] = {}
        for finding in self.blocking:
            unique.setdefault((finding.code, finding.message), finding)
        object.__setattr__(self, "blocking", tuple(unique.values()))

    @property
    def ok(self) -> bool:
        return not self.blocking


def collect_blocked(*field_states: FieldState[object]) -> tuple[Finding, ...]:
    """An ``advice.blocked`` finding naming each ``Blocked`` field state's
    root cause, in the order given.

    ``Unavailable`` and ``Resolved`` states are not blocking on their
    own -- an advisor that could not tell, or a caller that never
    supplied a given field at all, is not the same as an upstream
    failure. Only genuine ``Blocked`` propagation reaches here.
    """
    return tuple(
        replace(_BLOCKED, message=state.root_cause())
        for state in field_states
        if isinstance(state, Blocked)
    )


def check_all(
    *field_states: FieldState[object],
    occupations: FieldState[OccupationsDecision] | None = None,
    magnetic: FieldState[MagneticConfigFacts] | None = None,
    electron_count: FieldState[ElectronCountDecision] | None = None,
    relax: FieldState[RelaxOptions | VcRelaxOptions] | None = None,
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
    ``magnetic``/``electron_count``/``relax``/``boundary``/``vdw``/
    ``functional``/``job``/``parallel``/``nbnd``/``purpose``
    additionally opt into the named cross-parameter rules this module
    implements; passing them here also folds them into the generic
    ``Blocked`` scan, so a caller does not need to repeat them in
    ``field_states`` as well.

    Each rule returns its finding or ``None``; a new rule is one more
    entry in ``rule_findings`` and one in ``FINDING_CATALOGUE``.
    """
    optional = tuple(
        s
        for s in (
            occupations,
            magnetic,
            electron_count,
            relax,
            boundary,
            vdw,
            functional,
            job,
            parallel,
            nbnd,
        )
        if s is not None
    )
    rule_findings = (
        _fixed_occupations_needs_integer_moment(
            occupations, magnetic, electron_count, purpose
        ),
        _fixed_occupations_needs_even_electron_count(
            occupations, magnetic, electron_count, purpose
        ),
        _nbnd_holds_every_electron(nbnd, occupations, magnetic, electron_count),
        _vc_relax_requires_bfgs_ion_dynamics(relax, purpose),
        _isolation_matches_written_cell(boundary, magnetic),
        _vc_relax_forbids_isolated_boundary(boundary, purpose),
        _slab_vc_relax_needs_2d_cell_dofree(boundary, relax, magnetic),
        _two_d_cell_dofree_needs_xy_frame(relax, magnetic),
        _cartesian_cell_dofree_needs_unrotated_cell(relax, magnetic),
        _dispersion_needs_parameterised_functional(vdw, functional),
        _fix_bottom_layers_requires_slab(relax, magnetic),
        _npool_must_divide_ntasks(job, parallel),
        _ndiag_grid_must_distribute_bands(job, parallel, nbnd),
    )
    return CheckReport(
        blocking=(
            *collect_blocked(*field_states, *optional),
            *(finding for finding in rule_findings if finding is not None),
        )
    )


def _fixed_occupations_needs_integer_moment(
    occupations: FieldState[OccupationsDecision] | None,
    magnetic: FieldState[MagneticConfigFacts] | None,
    electron_count: FieldState[ElectronCountDecision] | None,
    purpose: str,
) -> Finding | None:
    if purpose not in SCF_LIKE_PURPOSES or occupations is None or magnetic is None:
        return None
    if not (occupations.ok and magnetic.ok):
        return None  # already surfaced via collect_blocked, or genuinely unknown
    if (
        occupations.value.occupations != "fixed"
        or magnetic.value.spin_mode != "collinear"
    ):
        return None
    tot = occupations.value.tot_magnetization
    if tot is None:
        return _MISSING_TOT_MAGNETIZATION
    if tot != int(tot):
        return replace(
            _NON_INTEGER_TOT_MAGNETIZATION,
            message=(
                "occupations=fixed on a collinear spin-polarized run needs an "
                f"integer tot_magnetization, not {tot:.10g}. Set "
                "occupations=smearing, or an integer tot_magnetization of the "
                "electron count's parity."
            ),
        )
    # nelup = (nelec + tot) / 2, neldw = (nelec - tot) / 2 (#70): both need
    # to be integers, which needs tot_magnetization and nelec to share
    # parity -- an integer tot_magnetization alone is not sufficient (e.g.
    # nelec=3, tot=2 gives nelup=2.5/neldw=0.5).
    if (
        electron_count is not None
        and electron_count.ok
        and not whole_spin_channels(electron_count.value.nelec, tot)
    ):
        return replace(
            _TOT_MAGNETIZATION_PARITY,
            message=(
                "occupations=fixed fills (nelec + tot_magnetization)/2 spin-up "
                "and (nelec - tot_magnetization)/2 spin-down electrons, which "
                "must both be whole numbers, so tot_magnetization must share "
                f"parity with the electron count; tot_magnetization={tot:.10g} "
                f"and nelec={electron_count.value.nelec:.10g} do not. Set "
                "occupations=smearing, or a tot_magnetization of the electron "
                "count's parity."
            ),
        )
    return None


def _fixed_occupations_needs_even_electron_count(
    occupations: FieldState[OccupationsDecision] | None,
    magnetic: FieldState[MagneticConfigFacts] | None,
    electron_count: FieldState[ElectronCountDecision] | None,
    purpose: str,
) -> Finding | None:
    """Sibling of ``_fixed_occupations_needs_integer_moment`` above, for
    runs without spin polarization, spin-orbit or not: found empirically
    2026-09-16, verified against real QE 7.3 source
    (``PW/src/iweights.f90``) -- see this module's own docstring, rule 3."""
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
    if not whole_spin_channels(nelec, 0.0):
        return replace(
            _ODD_ELECTRON_COUNT,
            message=(
                "occupations=fixed without spin polarization needs an even "
                f"electron count, but nelec={nelec:.10g} is odd, which leaves one "
                "band half filled. Set occupations=smearing, or "
                "spin_polarized=true with occupations=smearing for an unpaired "
                "electron."
            ),
        )
    return None


def _nbnd_holds_every_electron(
    nbnd: FieldState[NbndDecision] | None,
    occupations: FieldState[OccupationsDecision] | None,
    magnetic: FieldState[MagneticConfigFacts] | None,
    electron_count: FieldState[ElectronCountDecision] | None,
) -> Finding | None:
    """Rule 4 of this module's docstring: the first of pw.x's band floors
    ``nbnd`` falls below, with the fewest bands that pass them all."""
    if (
        nbnd is None
        or occupations is None
        or magnetic is None
        or electron_count is None
    ):
        return None
    if not (nbnd.ok and occupations.ok and magnetic.ok and electron_count.ok):
        return None  # already surfaced via collect_blocked, or genuinely unknown
    bands = nbnd.value.nbnd
    floors = band_floors(electron_count.value.nelec, occupations.value, magnetic)
    short = [floor for floor in floors if bands < floor.bands]
    if not short:
        return None
    floor = short[0]
    return replace(
        _TOO_FEW_BANDS,
        message=(
            f"nbnd={bands} cannot hold {floor.electrons:.10g} {floor.held}: pw.x "
            f"stops in setup with '{floor.error}'. Unset nbnd and extra_bands for "
            f"the default, or set nbnd to at least "
            f"{max(each.bands for each in floors)}."
        ),
    )


def _npool_must_divide_ntasks(
    job: FieldState[JobDecision] | None,
    parallel: FieldState[ParallelisationDecision] | None,
) -> Finding | None:
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
        return replace(
            _NPOOL,
            message=(
                f"npool={parallel.value.npool} does not divide "
                f"ntasks={job.value.ntasks}, and pw.x refuses that layout. Unset "
                "npool for the default, or set npool to a divisor of ntasks."
            ),
        )
    return None


def _ndiag_grid_must_distribute_bands(
    job: FieldState[JobDecision] | None,
    parallel: FieldState[ParallelisationDecision] | None,
    nbnd: FieldState[NbndDecision] | None,
) -> Finding | None:
    """Rule 5 of this module's docstring. ``parallelisation()`` never
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
    return replace(
        _NDIAG,
        message=(
            f"ndiag={ndiag} gives a {side}x{side} diagonalization grid, which "
            f"cannot distribute nbnd={bands} bands (it needs nbnd > "
            f"{(side - 1) ** 2}); pw.x would abort at startup or fail in the "
            "parallel eigensolver. Unset ndiag so pw.x chooses it from nbnd, "
            f"or use ndiag <= {(math.isqrt(bands - 1) + 1) ** 2}."
        ),
    )


def _vc_relax_requires_bfgs_ion_dynamics(
    relax: FieldState[RelaxOptions | VcRelaxOptions] | None,
    purpose: str,
) -> Finding | None:
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
        return replace(
            _VC_RELAX_ION_DYNAMICS,
            message=(
                "vc-relax is generated only with ion_dynamics=bfgs, not "
                f"{relax.value.ion_dynamics}: Quantum ESPRESSO pairs cell_dynamics "
                "with ion_dynamics in a vc-relax, and only the bfgs pair is "
                "written. Set ion_dynamics=bfgs."
            ),
        )
    return None


def _isolation_matches_written_cell(
    boundary: FieldState[BoundaryFacts] | None,
    magnetic: FieldState[MagneticConfigFacts] | None,
) -> Finding | None:
    """'isolated' and 'slab' truncate the Coulomb interaction, which is right
    only for the vacuum each assumes (#161). Checked on the cell the writer
    renders, whatever the source of ``isolation``."""
    if boundary is None or magnetic is None or not (boundary.ok and magnetic.ok):
        return None
    cell = magnetic.value.cell
    if boundary.value.isolation == "isolated":
        return _isolated_needs_vacuum_along_every_axis(cell)
    if boundary.value.isolation == "slab":
        return _slab_needs_room_for_the_2d_truncation(cell)
    return None


def _isolated_needs_vacuum_along_every_axis(cell: Structure) -> Finding | None:
    """Martyna-Tuckerman treats the system as a molecule or cluster in a 3D
    supercell (QE 7.3 PW/Doc/INPUT_PW.def:1593-1594); pw.x has no 1D
    truncation for a wire."""
    axes = vacuum_axes(cell)
    if len(axes) == 3:
        return None
    found = f"vacuum only along {_along(axes)}" if axes else "no vacuum"
    return replace(
        _ISOLATED_WITHOUT_VACUUM,
        message=(
            "isolation=isolated writes Martyna-Tuckerman (assume_isolated="
            "'martyna-tuckerman'), which treats the system as a molecule or "
            "cluster in a box and needs vacuum along a, b and c, but the cell to "
            f"be written has {found}. Set isolation=periodic."
        ),
    )


def _slab_needs_room_for_the_2d_truncation(cell: Structure) -> Finding | None:
    """QE's 2D Coulomb truncation cuts the interaction between charges more
    than half the cell height apart along z (QE 7.3
    PW/src/Coul_cut_2D.f90:77-81). It assumes a and b in the xy plane
    (Coul_cut_2D.f90:72-74, which only prints a warning) and c along z: its
    cutoff factor 1 - exp(-G_xy c/2) cos(G_z c/2) (Sohier et al., Phys. Rev.
    B 96, 075448 (2017), Eq. 24) is the Fourier transform of the truncated
    interaction only when every G_z is a multiple of 2 pi / c, so that the
    transform's sin(G_z c/2) term vanishes. It stops when the half height is
    below the pseudopotentials' radial cutoff, and truncates part of the
    slab's own interaction when the vacuum is too narrow;
    ``analysis.geometry.two_d_truncation_obstacles`` holds both bounds, and
    the boundary advisor offers 'slab' only where it finds nothing."""
    found = _xy_frame_obstacles(cell, allowed_vacuum=(("c",),))
    if not found:
        found = list(two_d_truncation_obstacles(cell))
    if not found:
        return None
    return replace(
        _SLAB_UNSUITED,
        message=(
            "isolation=slab writes Quantum ESPRESSO's 2D Coulomb truncation "
            "(assume_isolated='2D'), which needs vacuum along c only, a and b in "
            "the Cartesian xy plane, c along z, and room along z for the "
            "pseudopotentials and the slab, but the cell to be written has "
            f"{' and '.join(found)}. Set isolation=periodic."
        ),
    )


def _vc_relax_forbids_isolated_boundary(
    boundary: FieldState[BoundaryFacts] | None, purpose: str
) -> Finding | None:
    """Under Martyna-Tuckerman, from any source, pw.x clears ``tstress``
    (QE 7.3 PW/src/input.f90:1361-1365) even though vc-relax had switched
    it on (input.f90:646): BFGS then sees a zero cell gradient and reports
    the input cell as converged, exiting 0 (#126)."""
    if purpose != "vc-relax" or boundary is None or not boundary.ok:
        return None
    if boundary.value.isolation != "isolated":
        return None
    return _ISOLATED_VC_RELAX


def _slab_vc_relax_needs_2d_cell_dofree(
    boundary: FieldState[BoundaryFacts] | None,
    relax: FieldState[RelaxOptions | VcRelaxOptions] | None,
    magnetic: FieldState[MagneticConfigFacts] | None,
) -> Finding | None:
    """Under QE's 2D Coulomb truncation only in-plane stresses make sense, and
    QE asks a vc-relax to use cell_dofree='2Dxy' (QE 7.3
    PW/Doc/INPUT_PW.def:1677-1678); pw.x itself does not check it. Any value
    in ``_TWO_D_CELL_DOFREE`` holds c and the z components of a and b.
    Checked only on a written cell that is a slab with its vacuum along c in
    the xy frame, whose default is '2Dxy': there '2Dxy' passes the 2D-frame
    rule and is what isolation=periodic with cell_dofree unset writes, so
    both remedies hold and neither relaxes the vacuum. The boundary rule
    refuses isolation=slab on any other cell, with isolation=periodic as its
    remedy; there '2Dxy' fails the 2D-frame rule unless a and b lie in the xy
    plane without vacuum, and an unset cell_dofree blocks a wire or a slab
    with its vacuum along a or b and gives a box 'all'."""
    if boundary is None or relax is None or magnetic is None:
        return None
    if not (boundary.ok and relax.ok and magnetic.ok):
        return None
    options = relax.value
    if boundary.value.isolation != "slab" or not isinstance(options, VcRelaxOptions):
        return None
    if options.cell_dofree in _TWO_D_CELL_DOFREE:
        return None
    if _xy_frame_obstacles(magnetic.value.cell, allowed_vacuum=(("c",),)):
        return None
    return replace(
        _SLAB_VC_RELAX_CELL_DOFREE,
        message=(
            "vc-relax with isolation=slab cannot use "
            f"cell_dofree={options.cell_dofree}: under Quantum ESPRESSO's 2D "
            "Coulomb truncation only the in-plane stress is meaningful, so c and "
            "the z components of a and b must stay fixed. Set cell_dofree=2Dxy "
            "(or 2Dshape), or isolation=periodic with cell_dofree unset."
        ),
    )


def _two_d_cell_dofree_needs_xy_frame(
    relax: FieldState[RelaxOptions | VcRelaxOptions] | None,
    magnetic: FieldState[MagneticConfigFacts] | None,
) -> Finding | None:
    """QE's 2D cell constraints move only the x and y components of a and b
    and hold c fixed, in Cartesian axes (QE 7.3 Modules/cell_base.f90:742-749
    and :803-808), and BFGS judges convergence on the masked stress. With a
    or b out of the xy plane they distort any cell, bulk included: SCARF
    shrank a and b of rhombohedral Bi2Se3 under '2Dxy' while c stayed fixed.
    With vacuum along a or b they free a vacuum vector and freeze one in the
    sheet (#132). Checked on the cell the writer renders, whatever the
    source of ``cell_dofree``. 'all' is the remedy only for a cell without
    vacuum: with vacuum it would relax the vacuum too, which is why the relax
    advisor gives a wire or a slab outside the xy frame no default."""
    if relax is None or magnetic is None or not (relax.ok and magnetic.ok):
        return None
    options = relax.value
    if not isinstance(options, VcRelaxOptions):
        return None
    if options.cell_dofree not in _TWO_D_CELL_DOFREE:
        return None
    cell = magnetic.value.cell
    found = _xy_frame_obstacles(cell, allowed_vacuum=((), ("c",)))
    if not found:
        return None
    message = (
        f"cell_dofree={options.cell_dofree} moves only the x and y components "
        "of a and b and holds c fixed, so it needs a and b in the Cartesian xy "
        "plane and any vacuum along c, but the cell to be written has "
        f"{' and '.join(found)}."
    )
    axes = vacuum_axes(cell)
    if not axes:
        return replace(
            _TWO_D_CELL_DOFREE_FRAME,
            message=f"{message} Set cell_dofree=all, or reorient the cell.",
        )
    return replace(
        _TWO_D_CELL_DOFREE_FRAME,
        message=f"{message} {_holding_the_vacuum(axes)}",
        remedies=(),
    )


def _holding_the_vacuum(axes: tuple[LatticeAxis, ...]) -> str:
    """What holds the vacuum along ``axes`` fixed in a vc-relax, for a
    2D cell constraint refused on that cell: no orientation suits a wire or
    a box, whose two or three vacuum vectors cannot all lie along c."""
    if len(axes) == 3:
        return (
            "Use the relax task for a molecule or cluster in a box: every "
            "lattice vector is a vacuum vector, so a vc-relax would relax the "
            "vacuum."
        )
    if len(axes) == 2:
        period, constraint = wire_period(axes)
        return (
            "Set a cell_dofree that holds the vacuum fixed: "
            f"cell_dofree={constraint} holds {' and '.join(axes)} fixed and "
            f"relaxes only {period}, the wire's period, while cell_dofree=all "
            "would relax the vacuum too."
        )
    return (
        "Reorient the cell, or set a cell_dofree that holds the vacuum fixed: "
        "cell_dofree=all would relax the vacuum too."
    )


def _cartesian_cell_dofree_needs_unrotated_cell(
    relax: FieldState[RelaxOptions | VcRelaxOptions] | None,
    magnetic: FieldState[MagneticConfigFacts] | None,
) -> Finding | None:
    """Rule 6 of this module's docstring, whatever the source of
    ``cell_dofree``. The remedy clears the override rather than naming
    'all': a rotated cell is a bulk crystal, whose default is 'all', or a
    slab with its vacuum along c, whose default '2Dxy' keeps the vacuum
    fixed and also passes the 2D-truncation rule under isolation='slab',
    which 'all' would not."""
    if relax is None or magnetic is None or not (relax.ok and magnetic.ok):
        return None
    options = relax.value
    frame = magnetic.value.frame
    if not isinstance(options, VcRelaxOptions) or frame is None or not frame.rotated:
        return None
    if options.cell_dofree not in _CARTESIAN_CELL_DOFREE:
        return None
    return replace(
        _CARTESIAN_CELL_DOFREE_ROTATED,
        message=(
            f"cell_dofree={options.cell_dofree} holds or frees single Cartesian "
            "components of the lattice vectors, but the cell to be written is the "
            f"uploaded one {frame.describe_rotation()} into pw.x's frame, so its "
            "x, y and z are not the uploaded cell's. Unset cell_dofree for the "
            "default, or use cell_dofree=all or shape, or fixa, fixb or fixc to "
            "fix whole lattice vectors."
        ),
    )


def _xy_frame_obstacles(
    cell: Structure, allowed_vacuum: tuple[tuple[LatticeAxis, ...], ...]
) -> list[str]:
    """What keeps ``cell`` out of the frame QE's 2D methods assume: a or b out
    of the Cartesian xy plane, and vacuum axes other than those
    ``allowed_vacuum`` lists."""
    found = []
    if not ab_in_xy_plane(cell.lattice):
        found.append("a or b out of the xy plane")
    axes = vacuum_axes(cell)
    if axes not in allowed_vacuum:
        found.append(f"vacuum along {_along(axes)}" if axes else "no vacuum")
    return found


def _along(axes: tuple[LatticeAxis, ...]) -> str:
    """'c', 'a and b' or 'a, b and c'."""
    *rest, last = axes
    return f"{', '.join(rest)} and {last}" if rest else last


def _dispersion_needs_parameterised_functional(
    vdw: FieldState[VdwFacts] | None, functional: FieldState[str] | None
) -> Finding | None:
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
    return replace(
        _DISPERSION_FUNCTIONAL,
        message=(
            "use_vdw=true adds D3(BJ) dispersion, but Quantum ESPRESSO's DFT-D3 "
            f"has no parameters for functional {functional.value}, so pw.x would "
            "stop with 'functional name unknown'. Set use_vdw=false, or choose a "
            "functional with D3(BJ) parameters "
            f"({', '.join(sorted(D3BJ_FUNCTIONALS))})."
        ),
    )


def _fix_bottom_layers_requires_slab(
    relax: FieldState[RelaxOptions | VcRelaxOptions] | None,
    magnetic: FieldState[MagneticConfigFacts] | None,
) -> Finding | None:
    """``relax.fix_bottom_layers`` (#44) freezes a slab's bottom N atomic
    layers, which ``analysis.geometry.slab_layers`` counts up from the vacuum
    of a cell with exactly one vacuum axis. A '2d' rank is no evidence of a
    slab: bulk graphite ranks '2d' and has no vacuum (#160). Checked with the
    writer's own layers, on the cell it renders, so a passing check means the
    writer can apply the count."""
    if relax is None or magnetic is None or not (relax.ok and magnetic.ok):
        return None
    count = relax.value.fix_bottom_layers
    if count is None:
        return None
    cell = magnetic.value.cell
    layers = slab_layers(cell)
    if layers is None:
        return replace(
            _FIX_BOTTOM_LAYERS_WITHOUT_SLAB,
            message=(
                "fix_bottom_layers freezes the bottom atomic layers counted up "
                "from the vacuum, so it needs a slab: one vacuum gap, along "
                "exactly one lattice vector. The cell to be written has "
                f"{_no_single_slab(cell)}. Unset fix_bottom_layers."
            ),
        )
    if count > len(layers):
        return replace(
            _FIX_BOTTOM_LAYERS_EXCEEDS,
            message=(
                f"fix_bottom_layers={count} exceeds the {len(layers)} atomic "
                "layers of the slab to be written. Set fix_bottom_layers to at "
                f"most {len(layers)}, or unset it."
            ),
        )
    return None


def _no_single_slab(cell: Structure) -> str:
    """Why ``analysis.geometry.slab_layers`` finds no single slab in ``cell``:
    its vacuum axes, or a second vacuum gap along its one vacuum axis."""
    axes = vacuum_axes(cell)
    if len(axes) == 1:
        return f"more than one vacuum gap along {axes[0]}, so more than one slab"
    return f"vacuum along {_along(axes)}" if axes else "no vacuum"
