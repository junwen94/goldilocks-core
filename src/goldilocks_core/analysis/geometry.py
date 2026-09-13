"""geometry: dimensionality (3d/2d/1d/molecule), from a bonded-structure graph.

Ported as content from v1's ``legacy_analysis.py:197-220``
(``_analyze_dimensionality``) -- Larsen's connectivity-rank algorithm
(``pymatgen.analysis.dimensionality.get_dimensionality_larsen``) is reused
unchanged; only the bonding step changes. v1 built the bonded graph with
``CrystalNN``, a fingerprint-matching neighbour finder that raises
``ValueError``/``RuntimeError`` on disordered-adjacent or exotic-coordination
structures -- with nowhere for v1's ``analyze_structure`` to catch it, so the
whole calculation aborted (stfc/goldilocks-core#133). This swaps in
``JmolNN``, a simple cutoff-table bonding method with no fingerprint
matching to fail in the first place, and wraps whatever it still can't
resolve as ``Unavailable`` like every other ``analysis/`` fact -- not a
different, uncaught escape hatch (goldilocks-core-design.md:299-310's A7
asymmetry, the same fix as ``symmetry.py``).

This is a bonding-method swap plus tri-state, not a from-scratch
reimplementation of Larsen's graph algorithm -- that algorithm itself was
never the problem stfc/goldilocks-core#133 raised.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pymatgen.analysis.dimensionality import get_dimensionality_larsen
from pymatgen.analysis.local_env import JmolNN
from pymatgen.core import Structure
from pymatgen.core.graphs import StructureGraph

from goldilocks_core.resolution import FieldState, Provenance, Resolved, Unavailable

Dimensionality = Literal["3d", "2d", "1d", "molecule"]

_BY_VALUE: dict[int, Dimensionality] = {3: "3d", 2: "2d", 1: "1d", 0: "molecule"}


@dataclass(frozen=True, slots=True)
class GeometryFacts:
    dimensionality: Dimensionality
    low_dimensional: bool


def geometry(structure: Structure) -> FieldState[GeometryFacts]:
    if not structure.is_ordered:
        return Unavailable(
            reason="dimensionality classification does not support "
            "disordered structures"
        )
    try:
        bonded = StructureGraph.from_local_env_strategy(structure, JmolNN())
        dim_value = get_dimensionality_larsen(bonded)
    except (ValueError, RuntimeError) as error:
        return Unavailable(reason=f"dimensionality classification failed: {error}")
    dimensionality = _BY_VALUE.get(dim_value)
    if dimensionality is None:
        return Unavailable(
            reason=f"dimensionality classification returned an unrecognised "
            f"connectivity rank {dim_value!r}"
        )
    facts = GeometryFacts(
        dimensionality=dimensionality, low_dimensional=bool(dim_value < 3)
    )
    return Resolved(facts, Provenance(source="heuristic"))
