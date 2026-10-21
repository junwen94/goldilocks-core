"""boundary: how to handle a structure's electrostatic boundary conditions
in a periodic-cell code, from geometry.

New in v2 (v2 epic 5, #5): v1 computes `dimensionality` but nothing turns it
into a QE setting for the specific problem it causes -- a molecule or wire
placed in a periodic box picks up a spurious image-interaction energy offset
from its own periodic replicas (pitfall A8, goldilocks-core-design.md:4463).
`analysis/geometry.py`'s dimensionality/low_dimensional facts already exist;
this is their first real consumer for boundary-condition purposes.
`_advise_vdw` (v1's `advice/parameters.py`) already reads dimensionality for
a different purpose (dispersion correction, ported as `advisors/vdw_method.py`
in this same epic) -- so "dimensionality is a dangling output" was an
overstatement even before this epic; what was genuinely dangling is
specifically the boundary-condition settings this file now owns.

The decision is a code-neutral isolation category, not a Quantum ESPRESSO
keyword (#161): 'periodic' (no truncation), 'slab' (the Coulomb interaction
truncated along the slab normal) or 'isolated' (truncated along all three
lattice vectors). The Quantum ESPRESSO writer maps 'slab' to
``assume_isolated='2D'`` and 'isolated' to 'martyna-tuckerman', and writes
nothing for 'periodic'.

The heuristic decides from ``GeometryFacts.vacuum_axes``, not from the bond
rank. 'isolated' needs vacuum along all three lattice vectors: QE documents
Martyna-Tuckerman for "a molecule or a cluster in a 3D supercell" (QE 7.3
PW/Doc/INPUT_PW.def:1593-1594), while CO2 dry ice ranks 'molecule' but
fills its cell, and Martyna-Tuckerman would replace its periodic Hartree,
local-potential and Ewald terms with those of an isolated cluster (#126).
The method is exact only when the density fits in half the cell along every
axis (Martyna and Tuckerman, J. Chem. Phys. 110, 2810 (1999)), which a gap
wider than `analysis.geometry.VACUUM_GAP_ANGSTROM` does not guarantee.
pw.x computes no stress under it (QE 7.3 PW/src/input.f90:1361-1365), so
generation omits `tstress` and `checks.check_all` refuses a vc-relax.
Every other cell is 'periodic'. That includes wires, because pw.x has no 1D
truncation.

'slab' is QE's 2D Coulomb truncation (Sohier, Calandra and Mauri, Phys. Rev.
B 96, 075448 (2017)), which computes energies, forces and in-plane stresses
in a two-dimensional framework (QE 7.3 PW/Doc/INPUT_PW.def:1655-1678). It
comes only from a human or llm value: the heuristic keeps a slab 'periodic',
which is accurate for a neutral, symmetric monolayer with ~20 A of vacuum,
and no pw.x comparison of '2D' against periodic boundaries exists for the
bundled monolayers. ``checks.check_all`` refuses 'slab' and 'isolated', from
any source, on a written cell without the vacuum and frame they need.

A 2D slab's dipole correction (`tefield`/`dipfield`/`edir`/...) needs to
know whether the slab is symmetric, information this heuristic-tier fact
does not have. That is pitfall A9, explicitly deferred (see issue #5's
"Deferred from #2" list) rather than guessed at here: a '2d' rank gets a
warning instead of a guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from goldilocks_core.analysis.geometry import GeometryFacts
from goldilocks_core.inputs.overrides import HumanInput, LlmInput
from goldilocks_core.resolution import (
    Blocked,
    FieldState,
    Provenance,
    Resolved,
    Warning,
    blocked_by,
)

Isolation = Literal["periodic", "slab", "isolated"]
"""How the electrostatics treat the cell's periodic images: kept
('periodic'), cut along the slab normal ('slab') or cut along all three
lattice vectors ('isolated')."""

DIPOLE_CORRECTION_SUGGESTED = Warning(
    code="boundary.dipole_correction_suggested",
    level="info",
    category="boundary",
    message=(
        "2D structure in a periodic cell: consider a dipole correction "
        "(tefield/dipfield) if the slab is asymmetric along the vacuum "
        "direction -- not set automatically here."
    ),
)


WARNING_CATALOGUE = (DIPOLE_CORRECTION_SUGGESTED,)
"""Every warning code this module can emit -- ``capabilities.py``'s
``warnings[]`` catalogue aggregates one of these tuples per advisor."""


@dataclass(frozen=True, slots=True)
class BoundaryFacts:
    isolation: Isolation
    warnings: tuple[Warning, ...] = ()


class BoundaryHumanInput(HumanInput):
    isolation: Isolation | None = None


class BoundaryLlmInput(LlmInput):
    isolation: Isolation | None = None


def boundary(
    geometry: FieldState[GeometryFacts],
    human: BoundaryHumanInput | None = None,
    llm: BoundaryLlmInput | None = None,
) -> FieldState[BoundaryFacts]:
    """``human``/``llm`` (#36, v2 epic 9, #9): before this, ``boundary``
    was the only geometry-derived system decision with no override path
    at all -- unlike its sibling ``vdw_method`` (built in the same
    epic, reading the same ``GeometryFacts``), it had no escape hatch
    past a ``Blocked``/``Unavailable`` geometry, so a disordered/partial
    -occupancy structure (dimensionality classification does not
    support these) unconditionally blocked the whole run via this one
    field alone, even when every other setting the run needed was
    supplied. Matches ``vdw_method``'s own precedence: an explicit
    override wins outright, checked before ``geometry.ok``."""
    human = human or BoundaryHumanInput()
    llm = llm or BoundaryLlmInput()

    if human.isolation is not None:
        return Resolved(BoundaryFacts(human.isolation), Provenance(source="human"))
    ml_value: Isolation | None = None  # no ml model wired yet; epic 11's job
    if ml_value is not None:
        return Resolved(BoundaryFacts(ml_value), Provenance(source="ml"))
    if llm.isolation is not None:
        return Resolved(BoundaryFacts(llm.isolation), Provenance(source="llm"))

    if not geometry.ok:
        return Blocked(by=blocked_by(geometry))
    return _heuristic(geometry.value)


def _heuristic(facts: GeometryFacts) -> FieldState[BoundaryFacts]:
    if len(facts.vacuum_axes) == 3:
        return Resolved(
            BoundaryFacts("isolated"),
            Provenance(
                source="heuristic",
                source_note=(
                    "vacuum along all three lattice vectors: an isolated system "
                    "needs a Coulomb cutoff to avoid spurious interaction with "
                    "its own periodic images (pitfall A8)."
                ),
            ),
        )

    if facts.dimensionality == "2d":
        # Image interaction of a slab is normally handled by a dipole
        # correction -- deferred (pitfall A9), not guessed at here.
        return Resolved(
            BoundaryFacts("periodic", warnings=(DIPOLE_CORRECTION_SUGGESTED,)),
            Provenance(source="heuristic"),
        )
    return Resolved(BoundaryFacts("periodic"), Provenance(source="heuristic"))
