"""is_metal: metal | non_metal, from composition alone.

Fixes the bug in stfc/goldilocks-core#175 (v2 epic 4, #1): v1's
``heuristic_metallicity`` (``legacy_analysis.py:83-93``) requires *every*
element -- including anion-forming nonmetals like O/N -- to be metallic
(``all(element.is_metal for element in periodic_elements)``), so
RuO2/ReO3/LaNiO3/CrO2/TiN (all real metals) fall through to "unknown".
Common anion-forming nonmetals are excluded before checking metallicity of
what is left: those elements never carry the sign of whether the compound
conducts, the remaining (cation-side) elements do.

Still conservative like v1: this only ever confidently asserts "metal" from
composition alone (composition can't rule out a metal that merely looks
insulator-like, e.g. a narrow-gap semiconductor) -- anything else is
``Unavailable``, not a guessed "non_metal".
"""

from __future__ import annotations

from typing import Literal

from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element

from goldilocks_core.inputs.overrides import HumanInput, LlmInput
from goldilocks_core.resolution import FieldState, Provenance, Resolved, Unavailable

Metallicity = Literal["metal", "non_metal"]

_COMMON_ANIONS = frozenset({"O", "N", "S", "Se", "Te", "F", "Cl", "Br", "I", "H"})
"""Elements that commonly form anions and therefore say nothing about
whether the compound as a whole conducts -- the cation-side elements decide
that."""


class IsMetalHumanInput(HumanInput):
    is_metal: bool | None = None


class IsMetalLlmInput(LlmInput):
    is_metal: bool | None = None


def is_metal(
    structure: Structure,
    human: IsMetalHumanInput | None = None,
    llm: IsMetalLlmInput | None = None,
) -> FieldState[Metallicity]:
    human = human or IsMetalHumanInput()
    llm = llm or IsMetalLlmInput()
    if human.is_metal is not None:
        return Resolved(
            "metal" if human.is_metal else "non_metal", Provenance(source="human")
        )
    ml_value: bool | None = None  # no ml model wired yet; stubbed until epic 11
    if ml_value is not None:
        return Resolved("metal" if ml_value else "non_metal", Provenance(source="ml"))
    if llm.is_metal is not None:
        return Resolved(
            "metal" if llm.is_metal else "non_metal", Provenance(source="llm")
        )
    return _heuristic(structure)


def _heuristic(structure: Structure) -> FieldState[Metallicity]:
    elements = tuple(sorted(e.symbol for e in structure.composition.elements))
    cations = tuple(symbol for symbol in elements if symbol not in _COMMON_ANIONS)
    if cations and all(Element(symbol).is_metal for symbol in cations):
        return Resolved("metal", Provenance(source="heuristic"))
    return Unavailable(reason="composition alone does not confirm metallic character")
