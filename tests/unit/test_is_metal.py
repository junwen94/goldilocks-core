from __future__ import annotations

import pytest
from pymatgen.core import Lattice, Structure

from goldilocks_core.analysis.is_metal import (
    IsMetalHumanInput,
    IsMetalLlmInput,
    is_metal,
)


def _structure(*species: str) -> Structure:
    lattice = Lattice.cubic(4.0)
    coords = [[i / len(species), 0.0, 0.0] for i in range(len(species))]
    return Structure(lattice, species, coords)


@pytest.mark.parametrize(
    "species",
    [
        ("Ru", "O", "O"),  # RuO2
        ("Re", "O", "O", "O"),  # ReO3
        ("Ti", "N"),  # TiN
        ("La", "Ni", "O", "O", "O"),  # LaNiO3
        ("Cr", "O", "O"),  # CrO2
    ],
)
def test_oxides_and_nitrides_of_real_metals_resolve_as_metal(
    species: tuple[str, ...],
) -> None:
    """stfc/goldilocks-core#175: v1's all-elements-must-be-metallic rule
    misclassified every one of these as "unknown" because O/N are not
    metallic themselves -- fixed by excluding common anions first."""
    state = is_metal(_structure(*species))

    assert state.ok
    assert state.value == "metal"
    assert state.source == "heuristic"


def test_all_metallic_elements_resolve_as_metal() -> None:
    state = is_metal(_structure("Al"))

    assert state.value == "metal"


def test_metalloid_oxide_is_unavailable_not_a_guessed_non_metal() -> None:
    """Heuristic tier never confidently asserts non_metal from composition
    alone -- SiO2's cation (Si) is not metallic, so this can't confirm
    metallic character, but that is not evidence it is an insulator either."""
    silica = _structure("Si", "O", "O")

    state = is_metal(silica)

    assert not state.ok
    assert state.status == "unavailable"


def test_human_override_wins_over_heuristic() -> None:
    silica = _structure("Si", "O", "O")

    state = is_metal(silica, human=IsMetalHumanInput(is_metal=True))

    assert state.ok
    assert state.value == "metal"
    assert state.source == "human"


def test_llm_override_is_used_when_no_human_override_is_given() -> None:
    silica = _structure("Si", "O", "O")

    state = is_metal(silica, llm=IsMetalLlmInput(is_metal=False))

    assert state.ok
    assert state.value == "non_metal"
    assert state.source == "llm"


def test_human_override_takes_priority_over_llm_override() -> None:
    silica = _structure("Si", "O", "O")

    state = is_metal(
        silica,
        human=IsMetalHumanInput(is_metal=True),
        llm=IsMetalLlmInput(is_metal=False),
    )

    assert state.value == "metal"
    assert state.source == "human"
