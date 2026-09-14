from __future__ import annotations

import pytest
from pymatgen.core import Lattice, Structure

from goldilocks_core.advisors.functional import HYBRID_FUNCTIONAL
from goldilocks_core.advisors.hubbard_u import (
    CalibrationRequest,
    HubbardUHumanInput,
    expand_hubbard_label,
    hubbard_u,
)
from goldilocks_core.analysis.composition import composition
from goldilocks_core.analysis.needs_correlation import needs_correlation
from goldilocks_core.resolution import Blocked


def _structure(*species: str) -> Structure:
    lattice = Lattice.cubic(4.0)
    coords = [[i / len(species), 0.0, 0.0] for i in range(len(species))]
    return Structure(lattice, species, coords)


def test_metallic_transition_metal_does_not_get_plus_u() -> None:
    """stfc/goldilocks-core#175's "metallic Fe gets U=5.3" bug: no anion,
    no correction."""
    iron = _structure("Fe")

    state = hubbard_u(iron, needs_correlation(composition(iron)), "PBEsol")

    assert state.ok
    assert state.value.plan == "not_needed"


def test_forced_needs_correlation_on_a_non_correlated_structure_is_not_needed() -> None:
    """Regression for #36 (v2 epic 9, #9): forcing needs_correlation=True
    on a structure with zero transition-metal/lanthanide/actinide
    content (e.g. plain Si) used to fall through to plan='table' with
    an EMPTY u_by_element plus two Hubbard-specific warnings that make
    no sense with no +U term present -- and since write_qe_scf treats
    any plan != 'not_needed' as a real +U resolution, this failed with
    a factually wrong 'a Hubbard +U correction was resolved' error."""
    silicon = _structure("Si")

    state = hubbard_u(
        silicon,
        needs_correlation(composition(silicon)),
        "PBEsol",
        human=HubbardUHumanInput(needs_correlation=True),
    )

    assert state.ok
    assert state.value.plan == "not_needed"
    assert state.value.u_by_element == {}
    assert state.value.warnings == ()


def test_common_3d_oxide_uses_the_package_default_table() -> None:
    nio = _structure("Ni", "O")

    state = hubbard_u(nio, needs_correlation(composition(nio)), "PBEsol")

    assert state.value.plan == "table"
    assert state.value.u_by_element == {"Ni": 6.2}
    assert any(
        "projector" in warning.message.lower() for warning in state.value.warnings
    )


@pytest.mark.parametrize(
    ("species", "expected_element"),
    [
        (("Fe", "Fe", "O", "O", "O"), "Fe"),  # Fe2O3
        (("Co", "O"), "Co"),  # CoO
        (("Mn", "O"), "Mn"),  # MnO
    ],
)
def test_canonical_mott_insulators_get_plus_u(
    species: tuple[str, ...], expected_element: str
) -> None:
    """stfc/goldilocks-core#175: NiO/Fe2O3/CoO/MnO are the textbook
    O-bearing transition-metal Mott insulators GGA/LDA get wrong without
    +U -- all four must trigger a correction, not just NiO."""
    oxide = _structure(*species)

    state = hubbard_u(oxide, needs_correlation(composition(oxide)), "PBEsol")

    assert state.value.plan == "table"
    assert expected_element in state.value.u_by_element


def test_4d_transition_metal_oxide_recommends_calibration() -> None:
    """RuO2 (Ru is 4d) has no package-default U -- must recommend
    self-consistent calibration rather than guess."""
    ruo2 = _structure("Ru", "O", "O")

    state = hubbard_u(ruo2, needs_correlation(composition(ruo2)), "PBEsol")

    assert state.value.plan == "self_consistent_calibration_needed"
    assert state.value.calibration_requests == (
        CalibrationRequest(element="Ru", manifold="4d"),
    )


def test_lanthanide_oxide_recommends_calibration_with_4f_manifold() -> None:
    ceo2 = _structure("Ce", "O", "O")

    state = hubbard_u(ceo2, needs_correlation(composition(ceo2)), "PBEsol")

    assert state.value.plan == "self_consistent_calibration_needed"
    assert state.value.calibration_requests[0].element == "Ce"
    assert state.value.calibration_requests[0].manifold == "4f"
    assert state.value.calibration_requests[0].suggested_method == "hp.x"


def test_plus_u_from_the_table_warns_about_the_missing_initial_guess() -> None:
    """A10 (goldilocks-core-design.md:4465): +U has multiple metastable
    occupation-matrix solutions; this package does not compute a real
    starting_ns_eigenvalue guess, and must say so rather than stay silent."""
    nio = _structure("Ni", "O")

    state = hubbard_u(nio, needs_correlation(composition(nio)), "PBEsol")

    assert any(
        "starting_ns_eigenvalue" in warning.message for warning in state.value.warnings
    )


def test_calibration_needed_with_a_partial_table_still_warns_about_the_guess() -> None:
    """La2NiO4-shaped case: Ni is in the table (needs the A10 warning), Ru
    is not (needs calibration) -- mixing the two must not lose the warning
    for the element that IS resolved via the table."""
    mixed = _structure("Ni", "Ru", "O", "O", "O")

    state = hubbard_u(mixed, needs_correlation(composition(mixed)), "PBEsol")

    assert state.value.plan == "self_consistent_calibration_needed"
    assert state.value.u_by_element == {"Ni": 6.2}
    assert any(
        "starting_ns_eigenvalue" in warning.message for warning in state.value.warnings
    )


def test_hybrid_functional_suppresses_plus_u_regardless_of_composition() -> None:
    """goldilocks-core-design.md:2626-2627's cross-parameter rule."""
    nio = _structure("Ni", "O")

    state = hubbard_u(nio, needs_correlation(composition(nio)), HYBRID_FUNCTIONAL)

    assert state.value.plan == "not_needed"


def test_blocked_needs_correlation_propagates() -> None:
    iron = _structure("Fe")

    state = hubbard_u(iron, Blocked(by="synthetic failure"), "PBEsol")

    assert not state.ok
    assert state.status == "blocked"
    assert state.root_cause() == "synthetic failure"


def test_human_can_force_off() -> None:
    nio = _structure("Ni", "O")

    state = hubbard_u(
        nio,
        needs_correlation(composition(nio)),
        "PBEsol",
        human=HubbardUHumanInput(needs_correlation=False),
    )

    assert state.value.plan == "not_needed"
    assert state.source == "human"


def test_human_can_supply_an_explicit_u_table() -> None:
    ruo2 = _structure("Ru", "O", "O")

    state = hubbard_u(
        ruo2,
        needs_correlation(composition(ruo2)),
        "PBEsol",
        human=HubbardUHumanInput(u_by_element={"Ru": 4.0}),
    )

    assert state.value.plan == "table"
    assert state.value.u_by_element == {"Ru": 4.0}
    assert state.source == "human"


class TestExpandHubbardLabel:
    def test_expands_split_species_to_the_canonical_element_u_value(self) -> None:
        relabeled = Structure(
            Lattice.cubic(4.0),
            ["Fe", "Fe", "O", "O"],
            [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
            labels=["Fe1", "Fe2", "O", "O"],
        )

        expanded = expand_hubbard_label({"Fe": 5.3}, relabeled)

        assert expanded == {"Fe": 5.3, "Fe1": 5.3, "Fe2": 5.3}

    def test_does_not_override_an_already_individually_specified_split_species(
        self,
    ) -> None:
        relabeled = Structure(
            Lattice.cubic(4.0),
            ["Fe", "Fe"],
            [[0, 0, 0], [0.5, 0.5, 0.5]],
            labels=["Fe1", "Fe2"],
        )

        expanded = expand_hubbard_label({"Fe": 5.3, "Fe1": 6.0}, relabeled)

        assert expanded == {"Fe": 5.3, "Fe1": 6.0, "Fe2": 5.3}

    def test_identity_structure_with_no_split_species_is_unchanged(self) -> None:
        plain = _structure("Fe", "O")

        expanded = expand_hubbard_label({"Fe": 5.3}, plain)

        assert expanded == {"Fe": 5.3}
