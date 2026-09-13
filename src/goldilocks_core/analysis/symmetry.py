"""symmetry: space group and crystal system, via spglib.

Ported as content from v1's ``legacy_analysis.py:223-235``
(``_analyze_symmetry``) -- the ``SpacegroupAnalyzer`` classification itself
is reused unchanged. What changes is the failure handling: v1's
``analyze_structure`` caught ``SymmetryAnalysisError`` and downgraded every
field to ``None`` plus a warning string. That is exactly the ``Unavailable``
case, now named as one -- not an escape hatch peculiar to this one field
while an equivalent failure in ``geometry.py`` aborted the entire call
(goldilocks-core-design.md:299-310).

No ``human``/``llm`` override here: nothing in the evidence for this epic
(stfc/goldilocks-core#133/#175/#177) asks for one, and space group is
directly computed from geometry, not a judgement call.
"""

from __future__ import annotations

from dataclasses import dataclass

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from goldilocks_core.resolution import FieldState, Provenance, Resolved, Unavailable


@dataclass(frozen=True, slots=True)
class SymmetryFacts:
    space_group_symbol: str
    space_group_number: int
    crystal_system: str


def symmetry(structure: Structure) -> FieldState[SymmetryFacts]:
    if not structure.is_ordered:
        return Unavailable(
            reason="symmetry analysis does not support disordered structures"
        )
    try:
        analyzer = SpacegroupAnalyzer(structure)
        facts = SymmetryFacts(
            space_group_symbol=analyzer.get_space_group_symbol(),
            space_group_number=analyzer.get_space_group_number(),
            crystal_system=analyzer.get_crystal_system(),
        )
    except (TypeError, ValueError) as error:
        return Unavailable(reason=f"symmetry analysis failed: {error}")
    return Resolved(facts, Provenance(source="heuristic"))
