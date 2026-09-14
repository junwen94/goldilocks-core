from __future__ import annotations

from pymatgen.core import Lattice, Structure

from goldilocks_core.advisors.electron_count import (
    ElectronCountHumanInput,
    electron_count,
)
from goldilocks_core.assets.pseudopotentials.upf import PseudoMetadata
from goldilocks_core.resolution import Blocked, Provenance, Resolved, Unavailable

_IRON_OXIDE = Structure(
    Lattice.cubic(4.0),
    ["Fe", "Fe", "O", "O", "O"],
    [[i / 5, 0, 0] for i in range(5)],
)


def _metadata(element: str, z_valence: float) -> PseudoMetadata:
    return PseudoMetadata(
        filepath=f"{element}.upf",
        filename=f"{element}.upf",
        header_format="UPF v2",
        element=element,
        z_valence=z_valence,
    )


def test_sums_z_valence_times_atom_count_over_the_whole_structure() -> None:
    pseudos = Resolved(
        (_metadata("Fe", 16.0), _metadata("O", 6.0)), Provenance(source="heuristic")
    )

    state = electron_count(_IRON_OXIDE, pseudos)

    assert state.ok
    assert state.value.nelec == 2 * 16.0 + 3 * 6.0


def test_missing_z_valence_for_a_present_element_blocks() -> None:
    pseudos = Resolved((_metadata("Fe", 16.0),), Provenance(source="heuristic"))

    state = electron_count(_IRON_OXIDE, pseudos)

    assert not state.ok
    assert state.status == "blocked"
    assert "O" in state.root_cause()


def test_blocked_pseudos_propagates_as_blocked() -> None:
    state = electron_count(_IRON_OXIDE, Blocked(by="pseudopotential selection failed"))

    assert not state.ok
    assert state.root_cause() == "pseudopotential selection failed"


def test_unavailable_pseudos_is_treated_as_blocked_not_a_default() -> None:
    state = electron_count(_IRON_OXIDE, Unavailable(reason="synthetic failure"))

    assert not state.ok
    assert state.status == "blocked"


def test_human_override_bypasses_pseudos_entirely() -> None:
    state = electron_count(
        _IRON_OXIDE,
        Blocked(by="irrelevant"),
        human=ElectronCountHumanInput(nelec=50.0),
    )

    assert state.ok
    assert state.value.nelec == 50.0
    assert state.source == "human"
