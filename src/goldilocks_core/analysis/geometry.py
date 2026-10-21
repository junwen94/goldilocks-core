"""geometry: dimensionality (3d/2d/1d/molecule) from a bond graph.

The dimensionality is Larsen's connectivity rank
(``pymatgen.analysis.dimensionality.get_dimensionality_larsen``) of
``bond_graph``, which joins two bonding rules:

- ``JmolNN`` bonds a pair closer than the sum of two fixed, covalent-scale
  radii plus 0.45 A. It replaced v1's ``CrystalNN``, whose fingerprint
  matching raised on exotic coordination (stfc/goldilocks-core#133), but
  its cutoffs miss ionic, metallic and long covalent contacts: alone it
  ranked rock-salt NaCl, bcc Na and phosphorene 'molecule' (#125).
- Each site is also bonded to its nearest-neighbour shell, every neighbour
  closer than 1.1 times that site's shortest contact (the rule of
  pymatgen's ``MinimumDistanceNN(tol=0.1)``), so no atom of a dense solid
  is left unbonded.

Adding edges merges components or closes periodic cycles and never removes
either, so the union can only keep or raise the rank JmolNN alone gives.
Molecular crystals and van der Waals layers keep their rank because each
atom's nearest shell lies inside its own molecule or layer.

Disordered structures and classifier failures are ``Unavailable``, like
every other ``analysis/`` fact.
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

_SHELL_TOLERANCE = 0.1
"""A site is bonded to every neighbour closer than (1 + 0.1) times its
shortest contact (``MinimumDistanceNN``'s default ``tol``)."""

_SHELL_SEARCH_RADIUS_ANGSTROM = 6.0
"""Radius searched for each site's nearest-neighbour shell. It holds the
whole shell of the sparsest elemental solid (bcc Cs: shortest contact
5.32 A, shell edge 5.85 A). A site with no neighbour inside it stays
unbonded; ``MinimumDistanceNN``'s 10 A default would bond a lone atom in a
7-9.5 A box to its own periodic images."""


@dataclass(frozen=True, slots=True)
class GeometryFacts:
    dimensionality: Dimensionality
    low_dimensional: bool


def bond_graph(structure: Structure) -> StructureGraph:
    """Bond graph of an ordered ``structure``: ``JmolNN``'s bonds plus, for
    every site, a bond to each neighbour closer than
    ``1 + _SHELL_TOLERANCE`` times the site's shortest contact within
    ``_SHELL_SEARCH_RADIUS_ANGSTROM``. A site with no neighbour within that
    radius adds no bonds."""
    graph = StructureGraph.from_local_env_strategy(structure, JmolNN())
    neighbours_by_site = structure.get_all_neighbors(_SHELL_SEARCH_RADIUS_ANGSTROM)
    for index, neighbours in enumerate(neighbours_by_site):
        if not neighbours:
            continue
        shortest = min(neighbour.nn_distance for neighbour in neighbours)
        shell_edge = (1 + _SHELL_TOLERANCE) * shortest
        for neighbour in neighbours:
            if neighbour.nn_distance < shell_edge:
                graph.add_edge(
                    index,
                    neighbour.index,
                    to_jimage=neighbour.image,
                    warn_duplicates=False,
                )
    return graph


def geometry(structure: Structure) -> FieldState[GeometryFacts]:
    if not structure.is_ordered:
        return Unavailable(
            reason="dimensionality classification does not support "
            "disordered structures"
        )
    try:
        dim_value = get_dimensionality_larsen(bond_graph(structure))
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
