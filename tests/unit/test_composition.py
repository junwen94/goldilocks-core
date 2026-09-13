from __future__ import annotations

from pymatgen.core import Lattice, Structure

from goldilocks_core.analysis.composition import composition


def _structure(*species: str) -> Structure:
    lattice = Lattice.cubic(4.0)
    coords = [[i / len(species), 0.0, 0.0] for i in range(len(species))]
    return Structure(lattice, species, coords)


def test_composition_classifies_elements_into_periodic_table_families() -> None:
    state = composition(_structure("Fe", "O", "O", "O"))

    assert state.ok
    assert state.value.elements == ("Fe", "O")
    assert state.value.transition_metals == ("Fe",)
    assert state.value.lanthanides == ()
    assert state.value.actinides == ()


def test_composition_finds_lanthanides_and_actinides() -> None:
    state = composition(_structure("La", "Ni", "O", "O", "O"))

    # La is both a lanthanide and a transition metal by pymatgen's own
    # Element predicates -- the two classifications are not exclusive.
    assert state.value.transition_metals == ("La", "Ni")
    assert state.value.lanthanides == ("La",)
    assert state.value.actinides == ()


def test_composition_always_resolves() -> None:
    state = composition(_structure("Si"))

    assert state.ok
    assert state.status == "resolved"
    assert state.source == "heuristic"
