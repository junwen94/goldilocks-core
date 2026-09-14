from __future__ import annotations

from pymatgen.core import Lattice, Structure

from goldilocks_core.advisors.magnetic_config import (
    MagneticConfigHumanInput,
    magnetic_config,
)
from goldilocks_core.advisors.nbnd import NbndHumanInput, NbndLlmInput, nbnd
from goldilocks_core.advisors.occupations import OccupationsDecision
from goldilocks_core.analysis.is_magnetic import is_magnetic
from goldilocks_core.resolution import Blocked, Provenance, Resolved

_IRON = Structure(Lattice.cubic(2.87), ["Fe"], [[0.0, 0.0, 0.0]])
_SILICON = Structure(Lattice.cubic(5.43), ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])

_FIXED = Resolved(
    OccupationsDecision(occupations="fixed", smearing_type=None, degauss=None),
    Provenance(source="heuristic"),
)
_SMEARING = Resolved(
    OccupationsDecision(occupations="smearing", smearing_type="cold", degauss=0.01),
    Provenance(source="heuristic"),
)


def test_insulator_gets_half_of_nelec() -> None:
    state = nbnd(20.0, _FIXED)

    assert state.ok
    assert state.value.nbnd == 10


def test_insulator_rounds_an_odd_electron_count_up() -> None:
    state = nbnd(21.0, _FIXED)

    assert state.value.nbnd == 11  # ceil(10.5)


def test_metal_gets_twenty_percent_more_when_that_exceeds_the_floor() -> None:
    # nelec=200 -> valence_bands=100; 1.2*100=120 beats 100+4=104.
    state = nbnd(200.0, _SMEARING)

    assert state.value.nbnd == 120


def test_metal_gets_at_least_four_more_when_twenty_percent_is_smaller() -> None:
    # nelec=10 -> valence_bands=5; 1.2*5=6, but 5+4=9 is the floor.
    state = nbnd(10.0, _SMEARING)

    assert state.value.nbnd == 9


def test_tetrahedra_opt_is_treated_as_metal_like() -> None:
    tetrahedra = Resolved(
        OccupationsDecision(
            occupations="tetrahedra_opt", smearing_type=None, degauss=None
        ),
        Provenance(source="human"),
    )

    state = nbnd(10.0, tetrahedra)

    assert state.value.nbnd == 9


def test_blocked_occupations_propagates_as_blocked() -> None:
    state = nbnd(20.0, Blocked(by="synthetic failure"))

    assert not state.ok
    assert state.status == "blocked"
    assert state.root_cause() == "synthetic failure"


def test_human_nbnd_overrides_everything() -> None:
    state = nbnd(20.0, Blocked(by="synthetic failure"), human=NbndHumanInput(nbnd=42))

    assert state.ok
    assert state.value.nbnd == 42
    assert state.source == "human"


def test_human_extra_bands_adds_to_the_computed_base() -> None:
    state = nbnd(20.0, _FIXED, human=NbndHumanInput(extra_bands=5))

    assert state.value.nbnd == 15
    assert state.source == "human"


def test_llm_extra_bands_used_when_no_human_override() -> None:
    state = nbnd(20.0, _FIXED, llm=NbndLlmInput(extra_bands=3))

    assert state.value.nbnd == 13
    assert state.source == "llm"


def test_spin_polarized_system_gets_an_informational_note() -> None:
    magnetic = magnetic_config(_IRON, is_magnetic(_IRON))

    state = nbnd(16.0, _FIXED, magnetic=magnetic)

    assert any(
        "not the number of bands" in warning.message
        for warning in state.value.warnings
    )


def test_non_magnetic_system_has_no_spin_note() -> None:
    magnetic = magnetic_config(_SILICON, is_magnetic(_SILICON))

    state = nbnd(8.0, _FIXED, magnetic=magnetic)

    assert state.value.warnings == ()


def test_noncollinear_insulator_uses_the_full_electron_count_not_half() -> None:
    """QE's noncollinear bands hold one electron each (a two-component
    spinor), not two -- doubling the nominal insulator formula relative
    to a comparable collinear run (v2 epic 9, #9)."""
    magnetic = magnetic_config(
        _SILICON,
        is_magnetic(_SILICON),
        human=MagneticConfigHumanInput(spin_orbit_coupling=True),
    )

    state = nbnd(20.0, _FIXED, magnetic=magnetic)

    assert state.value.nbnd == 20


def test_noncollinear_metal_pads_from_the_full_electron_count() -> None:
    # nelec=200 -> valence_bands=200 (noncollinear, not halved);
    # 1.2*200=240 beats 200+4=204.
    magnetic = magnetic_config(
        _SILICON,
        is_magnetic(_SILICON),
        human=MagneticConfigHumanInput(spin_orbit_coupling=True),
    )

    state = nbnd(200.0, _SMEARING, magnetic=magnetic)

    assert state.value.nbnd == 240


def test_noncollinear_gets_its_own_note_not_the_nspin_two_one() -> None:
    magnetic = magnetic_config(
        _IRON,
        is_magnetic(_IRON),
        human=MagneticConfigHumanInput(spin_orbit_coupling=True),
    )

    state = nbnd(16.0, _FIXED, magnetic=magnetic)

    assert any(
        "one electron" in warning.message for warning in state.value.warnings
    )
    assert not any(
        "not the number of bands" in warning.message
        for warning in state.value.warnings
    )
