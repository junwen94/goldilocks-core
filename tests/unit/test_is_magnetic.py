from __future__ import annotations

import pytest
from pymatgen.core import Lattice, Structure

from goldilocks_core.analysis.is_magnetic import (
    IsMagneticHumanInput,
    IsMagneticLlmInput,
    is_magnetic,
)


def _structure(*species: str) -> Structure:
    lattice = Lattice.cubic(4.0)
    coords = [[i / len(species), 0.0, 0.0] for i in range(len(species))]
    return Structure(lattice, species, coords)


@pytest.mark.parametrize(
    "species",
    [
        ("Zn", "O"),  # ZnO: Zn2+ is d10
        ("Ti", "O", "O"),  # TiO2: Ti4+ is d0
        ("Cu", "Cu", "O"),  # Cu2O: Cu+ is d10
        ("Sc", "Sc", "O", "O", "O"),  # Sc2O3: Sc3+ is d0
    ],
)
def test_closed_shell_transition_metal_oxides_resolve_as_non_magnetic(
    species: tuple[str, ...],
) -> None:
    """stfc/goldilocks-core#175: v1's transition-metal-present rule
    advises spin-polarized for these -- Zn2+/Ti4+/Cu+/Sc3+ are d10/d0/d10/d0,
    closed or empty d shells, no unpaired electrons to align."""
    state = is_magnetic(_structure(*species))

    assert state.ok
    assert state.value == "non_magnetic"
    assert state.source == "heuristic"


@pytest.mark.parametrize(
    "species",
    [
        ("Fe", "Fe", "O", "O", "O"),  # Fe2O3: Fe3+ is d5
        ("Ni", "O"),  # NiO: Ni2+ is d8
    ],
)
def test_open_shell_transition_metal_oxides_resolve_as_magnetic(
    species: tuple[str, ...],
) -> None:
    state = is_magnetic(_structure(*species))

    assert state.ok
    assert state.value == "magnetic"


def test_lanthanide_presence_is_magnetic_without_checking_oxidation_state() -> None:
    """Narrowing by d-electron count only applies to transition metals here
    -- #175's evidence base is all transition-metal oxides, not lanthanides,
    so La keeps the broad v1-style "present -> magnetic" rule."""
    lanthanum_nickelate = _structure("La", "Ni", "O", "O", "O")

    state = is_magnetic(lanthanum_nickelate)

    assert state.value == "magnetic"


def test_no_magnetic_candidate_elements_resolves_as_non_magnetic() -> None:
    state = is_magnetic(_structure("Si"))

    assert state.value == "non_magnetic"


def test_ambiguous_oxidation_state_guess_is_unavailable() -> None:
    """FeNi3 has no anion to balance charge against, so
    Composition.oxi_state_guesses() returns nothing -- this must not guess
    either answer."""
    permalloy_like = _structure("Fe", "Ni", "Ni", "Ni")

    state = is_magnetic(permalloy_like)

    assert not state.ok
    assert state.status == "unavailable"


def test_human_override_wins_over_heuristic() -> None:
    zinc_oxide = _structure("Zn", "O")

    state = is_magnetic(zinc_oxide, human=IsMagneticHumanInput(is_magnetic=True))

    assert state.value == "magnetic"
    assert state.source == "human"


def test_llm_override_is_used_when_no_human_override_is_given() -> None:
    iron_oxide = _structure("Fe", "Fe", "O", "O", "O")

    state = is_magnetic(iron_oxide, llm=IsMagneticLlmInput(is_magnetic=False))

    assert state.value == "non_magnetic"
    assert state.source == "llm"
