"""magnetic_config: spin-polarization, starting magnetic moments, and
spin-orbit coupling, combined -- not three independent decisions that can
silently override each other.

New in v2 (v2 epic 5, #5). Fixes two real bugs found in v1's pipeline
(`advice/parameters.py`'s `_advise_magnetism`/`_advise_spin_orbit`,
`generation/qe/scf.py`'s `_spin_lines`), demonstrated here on this fresh
standalone advisor rather than by touching v1's pipeline directly --
same policy as every prior epic in this rewrite (v1 stayed untouched
until cutover, v2 epic 9, which deleted it in commit d64f46b). The two
`tests/physics/test_dft_decisions.py` tests this issue names ("A2"/
"A2b") used to `xfail` against `compute()`, v1's own pipeline
end-to-end, because v1's `advice/parameters.py` and
`generation/qe/scf.py` genuinely never changed; now that v1 is gone,
they exercise `magnetic_config()` directly instead. What this file
actually proves is in `tests/unit/test_advisors_magnetic_config.py`,
against the same physical scenarios (bulk Fe, with and without SOC).

- **A2** (stfc/goldilocks-core#177, still open upstream): v1 emits
  `nspin = 2` with every `starting_magnetization` implicitly zero, so QE
  relaxes to the non-magnetic solution while the run finishes, converges, and
  reports nothing obviously wrong. `spin_polarized=True` here always comes
  with a non-empty `starting_magnetization`.
- **A2b** (found while hardening `physics/` for v2 epic 1, #2): v1's
  `_spin_lines` checks `spin_orbit.enabled` first and returns before ever
  looking at `magnetism.spin_polarized` -- so enabling SOC on a structure
  independently advised as magnetic silently produces a non-magnetic
  noncollinear run. Here, spin-orbit coupling and magnetism are two
  independent fields on one result; enabling SOC never clears
  `starting_magnetization`, and a magnetic, SOC-enabled result also carries
  `angle1`/`angle2` (the moment direction QE's noncollinear mode needs) --
  SOC without magnetism only happens when the structure is genuinely
  non-magnetic, not because one flag silently overrode the other.

Also corrects a stricter-than-QE constraint the design doc's own audit found
elsewhere in itself (goldilocks-qe-pw-parameter-audit.md P0 finding #2):
giving both `tot_magnetization` and `starting_magnetization` does not error
in QE (`INPUT_PW`'s "do not specify both" is a recommendation, not an
enforced constraint -- verified against QE 7.3's `PW/src/input.f90`). Both
are accepted here as independent, unrelated overrides; this advisor never
treats supplying both as a blocking error.

**Spin mode (#146).** ``MagneticConfigFacts.spin_mode`` is the spin
treatment pw.x runs, derived from the two switches: ``noncollinear``
whenever spin-orbit coupling is on (``noncolin``, which pw.x runs with
nspin=4 and lsda off, magnetic or not -- QE 7.3 ``PW/src/input.f90:714-736``),
otherwise ``collinear`` for a spin-polarized run (nspin=2), otherwise
``none``. The gate, the occupations advisor, the writer and the job's spin
layout read it. ``tot_magnetization`` belongs to the collinear mode only:
pw.x stops on it without nspin=2 (``input.f90:761-762``). An explicit
``spin_polarized=false`` drops a requested one with the ignored warning,
with or without spin-orbit coupling; otherwise spin-orbit coupling refuses
it (``Blocked``), since the starting magnetization alone sets a
noncollinear run's moments.

**Two steps, around pseudopotential selection (#128).** ``magnetic_cell``
decides what selection depends on, and the simulated cell: spin
polarization, spin-orbit coupling (selection reads it for the relativistic
requirement), and the magnetic ordering applied to the structure -- the
only enumlib call. ``magnetic_config`` then sets the starting moments on
that cell, scaled by the selected pseudopotentials' ``z_valence``. Each
runs once per calculation (``service/_magnetic.py``), and everything
downstream -- electron count, bands, k-points, convergence, resources and
the writer -- reads the one cell object, so no count can describe a
different cell from the one written.

**AFM species-splitting (v2 epic 9, #9)**: the cell is the input
`structure` unless a caller explicitly opts in with
`human.magnetic_ordering="afm"` -- opt-in, not a new default, for the same
reason SOC stays opt-in below (real cost: this calls out to
`pymatgen.analysis.magnetism.analyzer.MagneticStructureEnumerator`, which
shells out to the external `enumlib` executables and can take seconds, not
milliseconds; it runs on a worker process in a private directory, see
`advisors/magnetic_enumeration.py`, #136). Finding which magnetic
configuration is the true ground state needs comparing several SCF total
energies -- explicitly an agent-level job in the design doc
(goldilocks-core-design.md:2076), not something this stateless, single-call
heuristic tier can or should attempt. What this tier gives instead is a
*reasonable, deterministic* compensated ordering (the smallest
antiferromagnetic candidate the enumerator finds, which can be a different
cell from the input: NiO's 2-atom primitive cell becomes the 4-atom AFM-II
cell), useful as a real starting point for a human/agent to actually run
and compare against the plain ferromagnetic guess -- degrading back to the
FM identity pass-through, with
a warning naming why, whenever enumlib is missing, the structure exceeds a
small site-count ceiling, pymatgen cannot enumerate it, enumlib runs out of
time, or no compensated ordering exists at all. Any other enumeration error
propagates. See `_afm_candidates_or_reason` below.

SOC stays opt-in, matching v1's own policy: `needs_soc` only ever produces a
warning recommending it be considered, never enables it automatically --
enabling SOC changes cost and setup, which is a human's call.

**`starting_magnetization` follows aiida-quantumespresso's own convention**
(`aiida_quantumespresso.workflows.protocols.utils.get_magnetization`, used by
`PwBaseWorkChain.get_builder_from_protocol(..., spin_type=SpinType.COLLINEAR)`;
see also the "Magnetic configurations" tutorial), not an invented flat
fraction: QE's `starting_magnetization` is defined as a species' spin
polarization -- a fraction of that species' pseudopotential valence electron
count, in [-1, 1] -- so the physically-motivated value is
``target_moment_in_bohr_magnetons / z_valence``, not one constant for every
element. `_MAGNETIC_MOMENT_TARGETS` below is aiida-quantumespresso's own
per-element table verbatim (elements with a partially-occupied d/f shell get
an aggressive maximal-moment target -- 5 or 7 Bohr magnetons -- to reliably
break spin symmetry; everything else gets 0, which maps to the flat
`DEFAULT_MAGNETIZATION_FRACTION` below). Every element in the structure gets
an entry when spin-polarized, not just the transition-metal/lanthanide/
actinide `magnetic_elements` -- aiida does the same, so that no single
species sits at exactly zero and re-locks the spin symmetry the rest of the
structure is trying to break.

Computing the real, calibrated fraction needs `z_valences` (each element's
*chosen pseudopotential's* valence electron count -- not a textbook/nominal
count, real UPF files vary, e.g. whether semicore states are included).
`pseudo_selection.py` picks that pseudopotential, and picking it needs
`spin_orbit_enabled` from *this* file (for the relativistic requirement) --
which is why the decision is the two steps above: `magnetic_config` takes
`z_valences` once selection has happened. Without them (no table installed
yet), every element still gets `DEFAULT_MAGNETIZATION_FRACTION` (aiida's own
flat default, not this package's invention) rather than either guessing a
`z_valence` or leaving `starting_magnetization` empty -- the latter would
silently reopen A2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.linalg import LinAlgError
from pymatgen.core import Structure

from goldilocks_core.advisors.magnetic_enumeration import (
    enumerate_orderings,
    enumlib_installed,
)
from goldilocks_core.analysis.composition import composition
from goldilocks_core.analysis.is_magnetic import Magnetism
from goldilocks_core.inputs.overrides import HumanInput, LlmInput
from goldilocks_core.resolution import (
    Blocked,
    FieldState,
    Provenance,
    Resolved,
    Warning,
    blocked_by,
)

_MAX_SITES_FOR_AFM_ENUMERATION = 16
"""Combinatorial enumeration cost grows quickly with site count; this
matches the order of magnitude the design doc itself treats as the
practical ceiling for magnetic sublattice counts (QE's own 3-character
``ATOMIC_SPECIES`` label limit already makes ``>=10`` inequivalent
magnetic sites "extremely rare" by goldilocks-core-design.md's own F19
note) -- not a benchmarked number, just a guard against a multi-minute
call for a case this heuristic tier was never meant to serve."""

DEFAULT_MAGNETIZATION_FRACTION = 0.1
"""aiida-quantumespresso's own flat default for elements with no specific
moment target (`magnetization.yaml`'s `default_magnetization`) -- also the
fallback for every element when `z_valences` isn't available at all, so
`starting_magnetization` is never empty on a spin-polarized result (A2)."""

_MAGNETIC_MOMENT_TARGETS: dict[str, float] = {
    "Ac": 5,
    "Ce": 5,
    "Co": 5,
    "Cr": 5,
    "Dy": 7,
    "Er": 7,
    "Eu": 7,
    "Fe": 5,
    "Gd": 5,
    "Hf": 5,
    "Ho": 7,
    "Ir": 5,
    "La": 5,
    "Lu": 5,
    "Mn": 5,
    "Mo": 5,
    "Nb": 5,
    "Nd": 7,
    "Ni": 5,
    "Np": 5,
    "Os": 5,
    "Pa": 5,
    "Pm": 7,
    "Pr": 7,
    "Pt": 5,
    "Pu": 7,
    "Re": 5,
    "Rh": 5,
    "Ru": 5,
    "Sc": 5,
    "Sm": 7,
    "Ta": 5,
    "Tb": 7,
    "Tc": 5,
    "Th": 5,
    "Ti": 5,
    "Tm": 7,
    "U": 5,
    "V": 5,
    "W": 5,
    "Y": 5,
    "Zr": 5,
}
"""Target initial magnetic moment (Bohr magnetons) per element, for elements
with a partially-occupied d or f shell -- an aggressive "aim for the maximal
possible moment" guess to reliably break spin symmetry, not a physical
prediction (ported verbatim from aiida-quantumespresso's
`workflows/protocols/magnetization.yaml`, itself a per-element, not a
per-oxidation-state, table). Every element not listed here has no specific
target and falls back to `DEFAULT_MAGNETIZATION_FRACTION`; notably this
includes Cu/Zn (filled or near-filled d-shell as a neutral element) despite
both being `is_transition_metal` in `analysis/composition.py`."""

SpinMode = Literal["none", "collinear", "noncollinear"]

_NONCOLLINEAR_TOT_MAGNETIZATION = (
    "tot_magnetization is not valid with spin_orbit_coupling: a spin-orbit run "
    "is noncollinear, and pw.x accepts tot_magnetization only for a collinear "
    "spin-polarized run (nspin=2); drop tot_magnetization and set the moments "
    "with starting_magnetization, or turn spin_orbit_coupling off."
)

_MAGNETIZATION_IGNORED_WARNING = Warning(
    code="magnetic.magnetization_ignored_spin_polarized_false",
    level="warning",
    category="magnetic",
    message=(
        "tot_magnetization/starting_magnetization were given but "
        "spin_polarized is explicitly false; both are ignored."
    ),
)

_SOC_SUGGESTED_WARNING = Warning(
    code="magnetic.soc_suggested",
    level="info",
    category="magnetic",
    message=(
        "heavy, non-s-block elements are present; consider enabling "
        "spin_orbit_coupling (not enabled automatically -- it changes "
        "cost and setup)."
    ),
)

WARNING_CATALOGUE = (
    _SOC_SUGGESTED_WARNING,
    Warning(
        code="magnetic.metallicity_unknown_defaulted_non_magnetic",
        level="warning",
        category="magnetic",
        message="magnetism could not be determined; defaulted to non-magnetic.",
    ),
    Warning(
        code="magnetic.afm_ordering_unavailable",
        level="warning",
        category="magnetic",
        message=(
            "antiferromagnetic ordering was requested but could not be "
            "produced; kept ferromagnetic order."
        ),
    ),
    Warning(
        code="magnetic.afm_ordering_applied",
        level="info",
        category="magnetic",
        message=(
            "a compensated antiferromagnetic ordering was found and applied; "
            "the input is written for its cell, which can differ from the "
            "uploaded one."
        ),
    ),
    _MAGNETIZATION_IGNORED_WARNING,
)
"""Every warning code this module can emit -- ``capabilities.py``'s
``warnings[]`` catalogue aggregates one of these tuples per advisor. The
real, per-occurrence message for the second code names the actual
reason (built at its call site in ``_resolve_spin_polarized`` below);
this is the generic description, for a caller that has not seen it fire."""


@dataclass(frozen=True, slots=True)
class OrderingRecord:
    """The magnetic ordering asked for (``"fm"`` when
    ``magnetic_ordering`` is unset, which behaves the same) and the one
    applied: ``"afm"`` when an antiferromagnetic candidate replaced the
    input cell, otherwise ``"fm"`` -- the input cell kept (the FM identity
    pass-through, as for every spin-unpolarized run)."""

    requested: Literal["fm", "afm"]
    applied: Literal["fm", "afm"]


@dataclass(frozen=True, slots=True)
class MagneticCell:
    """The simulated cell and the spin decisions taken before
    pseudopotential selection (module docstring).

    ``structure`` is the input structure, or the applied antiferromagnetic
    candidate: possibly a different cell, its sites spin-decorated and
    labelled by sublattice (``Ni1``/``Ni2``). ``cell_in_input_basis`` is its
    lattice in the input's basis, rows being lattice vectors:
    ``structure.lattice.matrix == cell_in_input_basis @ input.lattice.matrix``.
    That is an integer supercell matrix for a primitive input, but
    fractional when enumeration reduced a conventional input to a smaller
    cell (determinant 1/2 for rock-salt's 8-atom cell).

    ``tot_magnetization`` is the human request admitted for this run: set
    only for a collinear spin-polarized run (module docstring, #146)."""

    structure: Structure
    spin_polarized: bool
    spin_orbit_enabled: bool
    magnetic_elements: tuple[str, ...]
    ordering: OrderingRecord
    cell_in_input_basis: tuple[tuple[float, ...], ...]
    tot_magnetization: float | None
    warnings: tuple[Warning, ...] = ()


@dataclass(frozen=True, slots=True)
class MagneticConfigFacts:
    """The magnetic decision the writer renders. ``cell``, ``ordering``,
    ``spin_polarized``, ``magnetic_elements``, ``tot_magnetization`` and
    ``spin_orbit_enabled`` are copied from the ``MagneticCell`` (``cell``
    is the same object as ``MagneticCell.structure``); ``warnings`` are
    the cell's plus this step's."""

    cell: Structure
    ordering: OrderingRecord
    spin_polarized: bool
    magnetic_elements: tuple[str, ...]
    starting_magnetization: dict[str, float] | None
    tot_magnetization: float | None
    spin_orbit_enabled: bool
    angle1: dict[str, float] | None
    angle2: dict[str, float] | None
    warnings: tuple[Warning, ...] = ()

    @property
    def spin_mode(self) -> SpinMode:
        """``noncollinear`` with spin-orbit coupling, otherwise
        ``collinear`` when spin-polarized, otherwise ``none`` (module
        docstring). Derived from the two switches, so it cannot disagree
        with them; a property, not a field, so records do not carry it."""
        if self.spin_orbit_enabled:
            return "noncollinear"
        return "collinear" if self.spin_polarized else "none"


class MagneticConfigHumanInput(HumanInput):
    spin_polarized: bool | None = None
    spin_orbit_coupling: bool | None = None
    tot_magnetization: float | None = None
    starting_magnetization: dict[str, float] | None = None
    """A final, already-QE-scale fraction per element -- not a moment in
    Bohr magnetons -- matching aiida-quantumespresso's "Using the parameters"
    path (direct `parameters['SYSTEM']['starting_magnetization']`), as
    opposed to its "Using the SpinType" path (a physical moment scaled by
    `z_valences`, which is what the heuristic tier below does)."""
    magnetic_ordering: Literal["fm", "afm"] | None = None
    """Opt-in only (default ``None`` behaves exactly like ``"fm"``): stays
    off by default because attempting ``"afm"`` calls out to an external
    process and can take seconds, and because which ordering is *actually*
    the ground state is not something this module can determine on its
    own -- see the module docstring."""


class MagneticConfigLlmInput(LlmInput):
    spin_polarized: bool | None = None
    spin_orbit_coupling: bool | None = None


def magnetic_cell(
    structure: Structure,
    is_magnetic: FieldState[Magnetism],
    human: MagneticConfigHumanInput | None = None,
    llm: MagneticConfigLlmInput | None = None,
) -> FieldState[MagneticCell]:
    """The simulated cell (module docstring): spin polarization,
    spin-orbit coupling, the admitted ``tot_magnetization``, and
    ``structure`` with the requested ordering applied -- the one
    enumeration of a calculation. ``Blocked`` when ``is_magnetic`` is,
    when a human ``tot_magnetization`` comes with spin-orbit coupling
    (#146), or when a human ``starting_magnetization`` names a label the
    cell does not have."""
    human = human or MagneticConfigHumanInput()
    llm = llm or MagneticConfigLlmInput()

    if isinstance(is_magnetic, Blocked):
        # Blocked (something upstream of is_magnetic failed) is not the same
        # as Unavailable (the heuristic tried and could not tell) -- only
        # the former leaves nothing safe to default to.
        return Blocked(by=is_magnetic)

    spin_polarized, source, warnings = _resolve_spin_polarized(is_magnetic, human, llm)
    spin_orbit_enabled = human.spin_orbit_coupling is True
    # A requested moment belongs to a spin-polarized run: an explicit
    # spin_polarized=false drops it (the warning above). With spin-orbit
    # coupling the run is noncollinear, where pw.x has none (#146).
    tot_magnetization = human.tot_magnetization if spin_polarized else None
    if tot_magnetization is not None and spin_orbit_enabled:
        return Blocked(by=_NONCOLLINEAR_TOT_MAGNETIZATION)
    magnetic_elements = magnetic_elements_in(structure) if spin_polarized else ()
    requested = human.magnetic_ordering or "fm"

    cell = structure
    if spin_polarized and requested == "afm":
        cell, ordering_warnings = _afm_cell(structure, magnetic_elements)
        warnings = (*warnings, *ordering_warnings)

    if spin_polarized and human.starting_magnetization is not None:
        invalid = _invalid_labels(human.starting_magnetization, cell)
        if invalid:
            return Blocked(by=invalid)

    facts = MagneticCell(
        structure=cell,
        spin_polarized=spin_polarized,
        spin_orbit_enabled=spin_orbit_enabled,
        magnetic_elements=magnetic_elements,
        ordering=OrderingRecord(
            requested=requested, applied="fm" if cell is structure else "afm"
        ),
        cell_in_input_basis=_cell_in_input_basis(cell, structure),
        tot_magnetization=tot_magnetization,
        warnings=warnings,
    )
    return Resolved(facts, Provenance(source=source))


def magnetic_config(
    cell: FieldState[MagneticCell],
    needs_soc: FieldState[bool] | None = None,
    z_valences: dict[str, float] | None = None,
    human: MagneticConfigHumanInput | None = None,
) -> FieldState[MagneticConfigFacts]:
    """The starting moments on ``cell`` -- a human
    ``starting_magnetization`` (its labels already checked against the cell
    by ``magnetic_cell``), or the heuristic fractions, scaled by the
    selected pseudopotentials' ``z_valences`` when known -- with the
    spin-orbit suggestion and the moment angles. Copies the cell's facts,
    with the cell's provenance; ``Blocked`` when the cell is not
    resolved."""
    human = human or MagneticConfigHumanInput()
    if not cell.ok:
        return Blocked(by=blocked_by(cell))
    facts = cell.value

    starting_magnetization = None
    if facts.spin_polarized:
        starting_magnetization = (
            dict(human.starting_magnetization)
            if human.starting_magnetization is not None
            else _starting_magnetization_by_label(facts.structure, z_valences)
        )

    warnings = facts.warnings
    needs_soc_resolved_true = needs_soc is not None and needs_soc.ok and needs_soc.value
    if not facts.spin_orbit_enabled and needs_soc_resolved_true:
        warnings = (*warnings, _SOC_SUGGESTED_WARNING)

    angle1 = angle2 = None
    if facts.spin_orbit_enabled and starting_magnetization is not None:
        angle1 = dict.fromkeys(starting_magnetization, 0.0)
        angle2 = dict.fromkeys(starting_magnetization, 0.0)

    config = MagneticConfigFacts(
        cell=facts.structure,
        ordering=facts.ordering,
        spin_polarized=facts.spin_polarized,
        magnetic_elements=facts.magnetic_elements,
        starting_magnetization=starting_magnetization,
        tot_magnetization=facts.tot_magnetization,
        spin_orbit_enabled=facts.spin_orbit_enabled,
        angle1=angle1,
        angle2=angle2,
        warnings=warnings,
    )
    return Resolved(config, cell.provenance)


def _cell_in_input_basis(
    cell: Structure, structure: Structure
) -> tuple[tuple[float, ...], ...]:
    """``cell``'s lattice vectors in ``structure``'s basis (rows):
    ``cell.lattice.matrix == M @ structure.lattice.matrix``."""
    matrix = np.linalg.solve(structure.lattice.matrix.T, cell.lattice.matrix.T).T
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _resolve_spin_polarized(
    is_magnetic: FieldState[Magnetism],
    human: MagneticConfigHumanInput,
    llm: MagneticConfigLlmInput,
) -> tuple[bool, str, tuple[Warning, ...]]:
    magnetization_given = (
        human.tot_magnetization is not None or human.starting_magnetization is not None
    )
    if human.spin_polarized is not None:
        if human.spin_polarized is False and magnetization_given:
            # #34 (v2 epic 9, #9): an explicit spin_polarized=False wins
            # (this is a human override, not the heuristic default), but
            # both magnetization fields would otherwise be silently
            # dropped a few lines below with no signal anything was
            # ignored -- warn instead.
            return False, "human", (_MAGNETIZATION_IGNORED_WARNING,)
        return human.spin_polarized, "human", ()
    if magnetization_given:
        # #34 (v2 epic 9, #9): a human giving tot_magnetization/
        # starting_magnetization with no spin_polarized override at all
        # is a clear, unambiguous request for a spin-polarized
        # calculation -- before this, both were silently dropped
        # whenever spin_polarized otherwise resolved False (e.g. the
        # heuristic default on a structure not flagged magnetic).
        return True, "human", ()
    ml_value: bool | None = None  # no ml model wired yet; stubbed until epic 11
    if ml_value is not None:
        return ml_value, "ml", ()
    if llm.spin_polarized is not None:
        return llm.spin_polarized, "llm", ()
    if is_magnetic.ok:
        return is_magnetic.value == "magnetic", "heuristic", ()
    return (
        False,
        "heuristic",
        (
            Warning(
                code="magnetic.metallicity_unknown_defaulted_non_magnetic",
                level="warning",
                category="magnetic",
                message=(
                    f"magnetism could not be determined ({is_magnetic.reason}); "
                    "defaulted to non-magnetic."
                ),
            ),
        ),
    )


def magnetic_elements_in(structure: Structure) -> tuple[str, ...]:
    """Every transition-metal/lanthanide/actinide symbol present, sorted.

    Public (not ``_``-prefixed) because #87's ``enumerate_magnetic_orderings``
    needs the same set a caller outside this module's own
    ``magnetic_config()`` flow -- listing candidates before ``spin_polarized``
    has even been resolved for that caller's own purposes."""
    facts = composition(structure).value
    candidates = {*facts.transition_metals, *facts.lanthanides, *facts.actinides}
    return tuple(sorted(candidates))


def _fraction_for(symbol: str, z_valences: dict[str, float] | None) -> float:
    target = _MAGNETIC_MOMENT_TARGETS.get(symbol, 0)
    if not target:
        return DEFAULT_MAGNETIZATION_FRACTION
    if z_valences is None or symbol not in z_valences:
        # The real fraction needs this element's chosen pseudopotential's
        # valence electron count; without it, fall back to the flat default
        # rather than guess a z_valence -- still non-zero (A2), just not the
        # calibrated value yet.
        return DEFAULT_MAGNETIZATION_FRACTION
    return target / z_valences[symbol]


def afm_unavailable_warning(reason: str) -> Warning:
    """Public (v2 #87): ``service._magnetic_orderings.list_magnetic_orderings``
    needs the same warning shape ``_afm_cell`` already uses
    below, to explain *why* a listing came back FM-only rather than
    silently looking like every structure has no compensated AFM
    ordering at all."""
    return Warning(
        code="magnetic.afm_ordering_unavailable",
        level="warning",
        category="magnetic",
        message=f"{reason}; kept ferromagnetic order.",
    )


def _afm_candidates_or_reason(
    structure: Structure, magnetic_elements: tuple[str, ...]
) -> tuple[tuple[Structure, ...], str | None]:
    """Return every antiferromagnetic-origin candidate pymatgen's own
    ``MagneticStructureEnumerator`` (itself backed by
    ``MagOrderingTransformation``'s symmetry-aware enumeration) finds for
    ``structure``, or an empty tuple plus the reason there is none.

    Shared by ``_afm_cell`` (picks the smallest candidate, a
    deterministic starting point) and ``enumerate_magnetic_orderings``
    (lists every candidate, for a caller -- e.g. an mMACE-based energy
    ranking, #87 -- that wants to compare more than one). The enumerator
    runs on a worker process in a private directory
    (``magnetic_enumeration.enumerate_orderings``, #136).

    Only the enumerator's own answers about the structure become a reason,
    each with a fixed wording that quotes no exception text: pymatgen's
    ``ValueError`` (enumlib found no ordering, or the enumerator refused the
    structure as disordered, non-collinear, non-magnetic or with too many
    magnetic sites) and enumlib's ``TimeoutError``. Anything else propagates
    -- an ``OSError`` from a scratch directory, numpy's ``LinAlgError`` (a
    ``ValueError`` subclass), a dead worker process -- because it is a
    fault, not an answer about this structure."""
    if len(structure) > _MAX_SITES_FOR_AFM_ENUMERATION:
        return (), (
            f"structure has {len(structure)} sites, over the "
            f"{_MAX_SITES_FOR_AFM_ENUMERATION}-site ceiling this "
            "heuristic enumerates within"
        )
    if not enumlib_installed():
        return (), (
            "antiferromagnetic ordering needs the enumlib executables on PATH "
            "(enum.x or multienum.x, plus makestr.x, makeStr.x or makeStr.py) "
            "and at least one was not found"
        )

    try:
        orderings = enumerate_orderings(
            structure,
            default_magmoms=dict.fromkeys(magnetic_elements, 5.0),
            strategies=("ferromagnetic", "antiferromagnetic"),
            truncate_by_symmetry=True,
            max_orderings=16,
        )
    except LinAlgError:
        raise
    except ValueError:
        return (), "pymatgen could not enumerate magnetic orderings for this structure"
    except TimeoutError:
        return (), "enumlib did not finish within its time limit"

    candidates = tuple(candidate for candidate, origin in orderings if origin == "afm")
    if not candidates:
        return (), "no compensated antiferromagnetic ordering was found"
    return candidates, None


def enumerate_magnetic_orderings(
    structure: Structure, magnetic_elements: tuple[str, ...]
) -> tuple[tuple[tuple[str, Structure], ...], str | None]:
    """List every magnetic-ordering candidate this heuristic tier can
    produce for ``structure``: the plain ferromagnetic identity (label
    ``"fm"``), plus every antiferromagnetic candidate
    ``_afm_candidates_or_reason`` finds, labeled ``"afm-<n>"`` in
    enumeration order and already relabeled by spin (``_label_by_spin``) so
    a caller can use any entry directly as a simulated cell.

    Unlike ``_afm_cell``, this never picks a winner -- it is
    the listing half of #87's "enumerate, then optionally rank with mMACE,
    then generate" split. Returns just the ``"fm"`` entry, with no
    antiferromagnetic candidates, whenever ``_afm_candidates_or_reason``
    gives a reason (oversized structure, missing enumlib, a structure
    pymatgen cannot enumerate, enumlib's time limit, or no compensated
    ordering found); any other enumeration error propagates.

    The second return value is ``_afm_candidates_or_reason``'s own reason
    whenever no AFM candidate was found, ``None`` otherwise -- surfaced
    so a caller (#87's ``list_magnetic_orderings``) can tell a real "no
    compensated ordering exists" result apart from "AFM enumeration
    silently could not run here" (e.g. missing enumlib), which otherwise
    look identical from the candidate list alone."""
    candidates: list[tuple[str, Structure]] = [("fm", structure)]
    afm_candidates, reason = _afm_candidates_or_reason(structure, magnetic_elements)
    for index, candidate in enumerate(afm_candidates, start=1):
        candidates.append((f"afm-{index}", _label_by_spin(candidate)))
    return tuple(candidates), reason


def _afm_cell(
    structure: Structure, magnetic_elements: tuple[str, ...]
) -> tuple[Structure, tuple[Warning, ...]]:
    """The cell of one genuine, compensated two-sublattice antiferromagnetic
    ordering (the smallest candidate, relabeled by spin), or ``structure``
    itself -- the FM identity pass-through -- with a warning naming why
    ``_afm_candidates_or_reason`` could not produce one."""
    candidates, reason = _afm_candidates_or_reason(structure, magnetic_elements)
    if reason is not None:
        return structure, (afm_unavailable_warning(reason),)

    winner = min(candidates, key=lambda candidate: candidate.num_sites)
    cell = _label_by_spin(winner)
    applied = Warning(
        code="magnetic.afm_ordering_applied",
        level="info",
        category="magnetic",
        message=(
            f"antiferromagnetic ordering applied: {cell.num_sites} sites, "
            f"{len(set(cell.labels))} distinct species (was "
            f"{structure.num_sites} sites, {len(set(structure.species))} "
            "species)."
        ),
    )
    return cell, (applied,)


def _label_by_spin(candidate: Structure) -> Structure:
    """Turn a spin-decorated ``MagneticStructureEnumerator`` candidate's
    per-site spin into QE-``ATOMIC_SPECIES``-shaped labels (``Fe1``/``Fe2``),
    per goldilocks-core-design.md's F1/F8 label-expansion rule -- an
    element with only one spin value among its sites keeps its plain
    symbol; one with more than one gets numbered suffixes, most-positive
    spin first, for a deterministic (not enumeration-order-dependent)
    result."""
    spins_by_element: dict[str, list[float]] = {}
    for site in candidate:
        spin = getattr(site.specie, "spin", None) or 0.0
        spins_by_element.setdefault(site.specie.symbol, [])
        if spin not in spins_by_element[site.specie.symbol]:
            spins_by_element[site.specie.symbol].append(spin)

    label_by_key: dict[tuple[str, float], str] = {}
    for element, spins in spins_by_element.items():
        if len(spins) == 1:
            label_by_key[(element, spins[0])] = element
            continue
        for index, spin in enumerate(sorted(spins, reverse=True), start=1):
            label_by_key[(element, spin)] = f"{element}{index}"

    labels = [
        label_by_key[(site.specie.symbol, getattr(site.specie, "spin", None) or 0.0)]
        for site in candidate
    ]
    return Structure(
        candidate.lattice,
        candidate.species,
        candidate.frac_coords,
        labels=labels,
        site_properties=candidate.site_properties,
    )


def _invalid_labels(override: dict[str, float], structure: Structure) -> str | None:
    """A human-supplied ``starting_magnetization`` is keyed by QE species
    *label* (``Fe0``/``Fe1``, whatever the real structure's labels are),
    not element symbol -- the same fact ``_starting_magnetization_by_label``
    already keys by (fixed for #32). Unlike that heuristic path, an
    override's keys come from outside this codebase, so they need
    validating here rather than trusted: a bare element symbol that
    happens not to match any real label (pymatgen's own CIF reader
    already produces ``Fe0``/``Fe1``, not ``Fe``, for the common case)
    would otherwise reach ``generation/quantum_espresso/scf.py``'s
    ``_magnetic_keywords`` as a bare ``KeyError``, not a clean error --
    see this issue's own history (#46). Auto-expanding a bare symbol to
    every matching label is deliberately not attempted: different labels
    of one element can legitimately want different values (an AFM
    ``Fe1``/``Fe2`` pair), so guessing would mask a genuine mistake
    rather than catch it."""
    valid_labels = {site.label for site in structure}
    invalid = sorted(set(override) - valid_labels)
    if not invalid:
        return None
    return (
        f"starting_magnetization has unknown site label(s) {invalid!r}; "
        f"this structure's real labels are {sorted(valid_labels)!r}"
    )


def starting_magnetization_for(
    structure: Structure, z_valences: dict[str, float] | None = None
) -> dict[str, float]:
    """Public wrapper around ``_starting_magnetization_by_label`` -- #87's
    listing tier needs the same per-label, sign-aware fractions this module
    computes internally, to hand a listed AFM candidate's bundle-generation
    ``overrides`` back to a caller without duplicating the aiida-derived
    heuristic outside this module."""
    return _starting_magnetization_by_label(structure, z_valences)


def _starting_magnetization_by_label(
    structure: Structure, z_valences: dict[str, float] | None
) -> dict[str, float]:
    """The heuristic default, keyed by ``site.label`` (the QE-species label
    ``generation/quantum_espresso/scf.py``'s ``species_index`` also keys by),
    never by ``site.specie.symbol`` -- fixes #32 (v2 epic 9, #9), a
    regression #27 introduced: ``species_index`` became label-keyed to make
    AFM species-splitting work, but this function, called for every
    spin-polarized structure (not only AFM-relabeled ones), was still
    element-keyed -- a bare ``KeyError`` the instant a structure's own
    per-site labels are not identical to its element symbols, which
    pymatgen's own CIF writer/reader already does by default (e.g.
    ``Fe0``/``Fe1``, not ``Fe``) for every bundled example structure.

    Signed by each site's own spin direction where one is set (only true
    once AFM relabeling has run -- see ``_label_by_spin``); every other
    caller's sites carry no spin decoration at all, so ``sign`` is always
    ``1.0`` and every site with the same element gets the same (positive)
    fraction, exactly reproducing the old element-keyed heuristic's values,
    just correctly indexed by label instead of by symbol."""
    result: dict[str, float] = {}
    for site in structure:
        label = site.label
        if label in result:
            continue
        spin = getattr(site.specie, "spin", None) or 0.0
        sign = -1.0 if spin < 0 else 1.0
        result[label] = sign * _fraction_for(site.specie.symbol, z_valences)
    return result
