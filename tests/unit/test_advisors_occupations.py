from __future__ import annotations

from pymatgen.core import Lattice, Structure

from goldilocks_core.advisors.magnetic_config import magnetic_config
from goldilocks_core.advisors.occupations import (
    OccupationsHumanInput,
    OccupationsLlmInput,
    occupations,
)
from goldilocks_core.analysis.is_magnetic import is_magnetic
from goldilocks_core.analysis.is_metal import is_metal
from goldilocks_core.resolution import Blocked, Provenance, Resolved, Unavailable

_IRON = Structure(Lattice.cubic(2.87), ["Fe"], [[0.0, 0.0, 0.0]])
_SILICON = Structure(Lattice.cubic(5.43), ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])


def test_metal_gets_cold_smearing() -> None:
    state = occupations(is_metal(_IRON))

    assert state.ok
    assert state.value.occupations == "smearing"
    assert state.value.smearing_type == "cold"
    assert state.value.degauss == 0.01


def test_smearing_type_and_degauss_apply_without_also_setting_occupations() -> None:
    """Regression for #34 (v2 epic 9, #9): a human-supplied smearing_type/
    degauss with no matching occupations override used to be silently
    dropped in favor of the hardcoded 'cold'/0.01 heuristic default,
    since only the explicit-occupations branch ever read them."""
    state = occupations(
        is_metal(_IRON),
        human=OccupationsHumanInput(smearing_type="methfessel-paxton", degauss=0.05),
    )

    assert state.ok
    assert state.value.occupations == "smearing"
    assert state.value.smearing_type == "methfessel-paxton"
    assert state.value.degauss == 0.05


def test_confirmed_non_metal_gets_fixed_occupations() -> None:
    state = occupations(Resolved("non_metal", Provenance(source="heuristic")))

    assert state.value.occupations == "fixed"
    assert state.value.smearing_type is None
    assert state.value.degauss is None


def test_unavailable_metallicity_defaults_to_smearing_not_blocked() -> None:
    """Unavailable means "could not confirm either way" -- since v2's
    is_metal is more conservative than v1's classification, this bucket
    now includes cases that look metallic but are not fully confirmed, so
    it gets the safe universal default (smearing), not a guessed fixed."""
    state = occupations(Unavailable(reason="composition alone does not confirm"))

    assert state.ok
    assert state.value.occupations == "smearing"


def test_blocked_metallicity_propagates_as_blocked() -> None:
    state = occupations(Blocked(by="synthetic failure"))

    assert not state.ok
    assert state.status == "blocked"
    assert state.root_cause() == "synthetic failure"


def test_human_can_force_smearing_with_custom_values() -> None:
    state = occupations(
        is_metal(_SILICON),
        human=OccupationsHumanInput(
            occupations="smearing", smearing_type="gaussian", degauss=0.02
        ),
    )

    assert state.value.occupations == "smearing"
    assert state.value.smearing_type == "gaussian"
    assert state.value.degauss == 0.02
    assert state.source == "human"


def test_human_can_force_tetrahedra_opt() -> None:
    state = occupations(
        is_metal(_IRON), human=OccupationsHumanInput(occupations="tetrahedra_opt")
    )

    assert state.value.occupations == "tetrahedra_opt"
    assert state.value.smearing_type is None
    assert state.value.degauss is None


def test_llm_can_choose_fixed_over_a_metal_heuristic() -> None:
    state = occupations(is_metal(_IRON), llm=OccupationsLlmInput(occupations="fixed"))

    assert state.value.occupations == "fixed"
    assert state.source == "llm"


def test_fixed_occupations_on_a_spin_polarized_system_warns() -> None:
    magnetic = magnetic_config(_IRON, is_magnetic(_IRON))

    state = occupations(Resolved("non_metal", Provenance(source="heuristic")), magnetic)

    assert state.value.occupations == "fixed"
    assert any(
        "integer tot_magnetization" in warning.message
        for warning in state.value.warnings
    )


def test_fixed_occupations_on_a_non_magnetic_system_has_no_warning() -> None:
    magnetic = magnetic_config(_SILICON, is_magnetic(_SILICON))

    state = occupations(Resolved("non_metal", Provenance(source="heuristic")), magnetic)

    assert state.value.warnings == ()


def test_smearing_never_gets_the_fixed_spin_warning() -> None:
    magnetic = magnetic_config(_IRON, is_magnetic(_IRON))

    state = occupations(is_metal(_IRON), magnetic)

    assert state.value.occupations == "smearing"
    assert state.value.warnings == ()
