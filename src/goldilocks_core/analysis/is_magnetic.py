"""is_magnetic: magnetic | non_magnetic, from composition and guessed
oxidation states.

Fixes the bug in stfc/goldilocks-core#175 (v2 epic 4, #1): v1 has no
``is_magnetic`` function -- the closest equivalent, ``advice/parameters.py``'s
``_advise_magnetism``, triggers on any transition-metal/lanthanide/actinide
*presence* with no regard to oxidation state, so ZnO/TiO2/Cu2O/Sc2O3 (Zn2+,
Ti4+, Cu+, Sc3+ -- d10, d0, d10, d0: closed or empty d shells, no unpaired d
electrons) get advised spin-polarised for nothing.

Lanthanides/actinides keep the broad "present -> magnetic" rule unchanged:
#175's evidence is all transition-metal oxides, and f-electron counting is a
materially different problem from the d-electron rule below -- narrowing
that is future work with no evidence base here, not something to guess at.

For transition metals, d-electron count = group number - oxidation state
(the standard crystal-field-theory convention for groups 3-12); d0 or d10
means a closed or empty d shell, hence no unpaired d electrons to align.
Oxidation states are guessed from composition alone
(``pymatgen.core.Composition.oxi_state_guesses`` -- ICSD-derived charge
statistics, no bond geometry, no network, no installed model). When the
guess is empty, ambiguous, or chemically implausible, this returns
``Unavailable`` rather than assuming either answer -- exactly the case
tri-state exists for.
"""

from __future__ import annotations

from typing import Literal

from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element

from goldilocks_core.inputs.overrides import HumanInput, LlmInput
from goldilocks_core.resolution import FieldState, Provenance, Resolved, Unavailable

Magnetism = Literal["magnetic", "non_magnetic"]

_CLOSED_SHELL_D_COUNTS = frozenset({0, 10})


class IsMagneticHumanInput(HumanInput):
    is_magnetic: bool | None = None


class IsMagneticLlmInput(LlmInput):
    is_magnetic: bool | None = None


def is_magnetic(
    structure: Structure,
    human: IsMagneticHumanInput | None = None,
    llm: IsMagneticLlmInput | None = None,
) -> FieldState[Magnetism]:
    human = human or IsMagneticHumanInput()
    llm = llm or IsMagneticLlmInput()
    if human.is_magnetic is not None:
        return Resolved(
            "magnetic" if human.is_magnetic else "non_magnetic",
            Provenance(source="human"),
        )
    ml_value = _ml_is_magnetic(structure)
    if ml_value is not None:
        return Resolved(
            "magnetic" if ml_value else "non_magnetic", Provenance(source="ml")
        )
    if llm.is_magnetic is not None:
        return Resolved(
            "magnetic" if llm.is_magnetic else "non_magnetic",
            Provenance(source="llm"),
        )
    return _heuristic(structure)


def _ml_is_magnetic(structure: Structure) -> bool | None:
    """The published mMACE-embedding MLP classifier (v2 epic 11, #11), or
    ``None`` if its model asset is not installed, goldilocks-ml's
    ``magnetism`` extra is not importable, or no mMACE backbone is
    configured (``GOLDILOCKS_MACE_BACKBONE``) -- never a reason to fail
    ``is_magnetic()`` itself, the same degrade-to-heuristic policy every
    ML-backed advisor in this codebase already follows for a missing
    external dependency."""
    from goldilocks_core.ml.predict import MlModelUnavailable, predict

    try:
        prediction = predict("is_magnetic", structure)
    except MlModelUnavailable:
        return None
    return bool(prediction.value)


def _heuristic(structure: Structure) -> FieldState[Magnetism]:
    elements = tuple(sorted(e.symbol for e in structure.composition.elements))
    periodic_elements = tuple(Element(symbol) for symbol in elements)

    if any(
        element.is_lanthanoid or element.is_actinoid for element in periodic_elements
    ):
        return Resolved("magnetic", Provenance(source="heuristic"))

    transition_metals = tuple(
        element for element in periodic_elements if element.is_transition_metal
    )
    if not transition_metals:
        return Resolved("non_magnetic", Provenance(source="heuristic"))

    try:
        guesses = structure.composition.oxi_state_guesses()
    except ValueError:
        # e.g. a disordered structure's fractional site occupancies give a
        # non-integer composition, which oxi_state_guesses refuses outright.
        guesses = ()
    if not guesses:
        return Unavailable(
            reason="could not guess oxidation states to check d-electron count"
        )
    best_guess = guesses[0]

    for element in transition_metals:
        oxidation_state = best_guess.get(element.symbol)
        if oxidation_state is None:
            return Unavailable(reason=f"oxidation state guess omits {element.symbol}")
        d_electrons = round(element.group - oxidation_state)
        if not 0 <= d_electrons <= 10:
            return Unavailable(
                reason=f"{element.symbol}{oxidation_state:+g} implies an "
                f"unphysical d-electron count ({d_electrons})"
            )
        if d_electrons not in _CLOSED_SHELL_D_COUNTS:
            return Resolved("magnetic", Provenance(source="heuristic"))

    return Resolved("non_magnetic", Provenance(source="heuristic"))
