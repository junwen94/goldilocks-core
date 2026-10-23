"""is_metal: whether the structure is metallic (zero DFT band gap).

Tiers, first resolved wins: a human override, the published CGCNN
classifier (``_ml_is_metal``), an LLM override, then the composition
heuristic (``_heuristic``).

The heuristic asserts ``metal`` only for a composition made entirely of
metallic elements by pymatgen's ``Element.is_metal`` (elements and
intermetallics). Any non-metal element leaves it ``Unavailable``: whether a
compound with non-metals conducts depends on its band filling, so
insulating NaCl, MgO and ZnO and metallic RuO2, ReO3, TiN, LaNiO3 and CrO2
look alike to composition (#164). This reverses stfc/goldilocks-core#175,
which stripped common anions and asked whether the remaining elements were
metals, and so called nearly every ionic insulator a metal. Generated
inputs do not change: occupations and k_sampling treat ``Unavailable``
like ``metal``.

The heuristic never guesses ``non_metal``: composition cannot rule out a
metal that looks insulator-like. An all-metal ``metal`` can still be wrong,
for intermetallic semiconductors such as CsAu or Mg2Sn.
"""

from __future__ import annotations

from typing import Literal

from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element

from goldilocks_core.inputs.overrides import HumanInput, LlmInput
from goldilocks_core.resolution import FieldState, Provenance, Resolved, Unavailable

Metallicity = Literal["metal", "non_metal"]


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
    ml_value = _ml_is_metal(structure)
    if ml_value is not None:
        return Resolved("metal" if ml_value else "non_metal", Provenance(source="ml"))
    if llm.is_metal is not None:
        return Resolved(
            "metal" if llm.is_metal else "non_metal", Provenance(source="llm")
        )
    return _heuristic(structure)


def _ml_is_metal(structure: Structure) -> bool | None:
    """The published CGCNN classifier (v2 epic 11, #11), or ``None`` if it
    is not set up (its model asset is not installed or fails verification),
    in which case ``is_metal()`` falls back to its next tier. A model that is
    set up but fails to load or predict raises, failing ``is_metal()``
    itself: a silent heuristic answer would change the generated input
    (#130)."""
    from goldilocks_core.ml.predict import MlModelUnavailable, predict

    try:
        prediction = predict("is_metal", structure)
    except MlModelUnavailable:
        return None
    return bool(prediction.value)


def _heuristic(structure: Structure) -> FieldState[Metallicity]:
    """``metal`` when every element is metallic, otherwise ``Unavailable``
    naming the non-metal elements."""
    elements = sorted({e.symbol for e in structure.composition.elements})
    non_metals = [symbol for symbol in elements if not Element(symbol).is_metal]
    if non_metals:
        listed = ", ".join(non_metals)
        return Unavailable(
            reason=f"composition contains non-metal elements ({listed}); whether "
            "such a compound conducts depends on its band filling, which "
            "composition alone cannot settle"
        )
    return Resolved("metal", Provenance(source="heuristic"))
