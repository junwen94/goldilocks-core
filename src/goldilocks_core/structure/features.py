from __future__ import annotations

from dataclasses import dataclass

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


@dataclass
class MatchFeatures:
    formula_reduced: str
    elements: list[str]
    nsites: int
    spacegroup_number: int
    spacegroup_symbol: str


def extract_match_features(structure: Structure) -> MatchFeatures:
    sga = SpacegroupAnalyzer(structure)
    composition = structure.composition
    return MatchFeatures(
        formula_reduced=composition.reduced_formula,
        elements=sorted(composition.chemical_system.split("-")),
        nsites=len(structure),
        spacegroup_number=sga.get_space_group_number(),
        spacegroup_symbol=sga.get_space_group_symbol(),
    )
