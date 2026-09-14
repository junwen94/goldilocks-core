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

Scope decision: `assume_isolated` only, not the full boundary-condition
picture. A 2D slab's periodic-image problem is normally solved with a dipole
correction (`tefield`/`dipfield`/`edir`/...) rather than `assume_isolated`,
and that correction needs to know which direction is the vacuum gap and
whether the slab is symmetric -- information this heuristic-tier fact does
not have. That is pitfall A9, explicitly deferred (see issue #5's "Deferred
from #2" list) rather than guessed at here. `assume_isolated` is set only for
0D (molecule) and 1D (wire) cases, where QE's Martyna-Tuckerman method
(a real-space Coulomb cutoff, not specific to any one reduced dimension)
applies directly; 2D gets a warning instead of a guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from goldilocks_core.analysis.geometry import GeometryFacts
from goldilocks_core.resolution import (
    Blocked,
    FieldState,
    Provenance,
    Resolved,
    Warning,
    blocked_by,
)

AssumeIsolated = Literal["none", "martyna-tuckerman"]

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
    assume_isolated: AssumeIsolated
    warnings: tuple[Warning, ...] = ()


def boundary(geometry: FieldState[GeometryFacts]) -> FieldState[BoundaryFacts]:
    if not geometry.ok:
        return Blocked(by=blocked_by(geometry))
    return _heuristic(geometry.value)


def _heuristic(facts: GeometryFacts) -> FieldState[BoundaryFacts]:
    if not facts.low_dimensional:
        return Resolved(BoundaryFacts("none"), Provenance(source="heuristic"))

    if facts.dimensionality in ("molecule", "1d"):
        return Resolved(
            BoundaryFacts("martyna-tuckerman"),
            Provenance(
                source="heuristic",
                source_note=(
                    f"{facts.dimensionality} structure in a periodic cell needs a "
                    "real-space Coulomb cutoff to avoid spurious interaction "
                    "with its own periodic images (pitfall A8)."
                ),
            ),
        )

    # 2D slabs: image interaction is normally handled by a dipole correction,
    # not assume_isolated -- deferred (pitfall A9), not guessed at here.
    return Resolved(
        BoundaryFacts("none", warnings=(DIPOLE_CORRECTION_SUGGESTED,)),
        Provenance(source="heuristic"),
    )
