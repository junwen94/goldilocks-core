from __future__ import annotations

from pymatgen.core import Lattice, Structure

from goldilocks_core.advisors.magnetic_config import (
    MagneticConfigHumanInput,
    magnetic_config,
)
from goldilocks_core.analysis.is_magnetic import is_magnetic
from goldilocks_core.resolution import Blocked, Resolved, Unavailable

_IRON = Structure(Lattice.cubic(2.87), ["Fe"], [[0.0, 0.0, 0.0]])
_SILICON = Structure(Lattice.cubic(5.43), ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])


def test_magnetic_structure_gets_spin_polarized_and_a_starting_magnetization() -> None:
    """A2 (stfc/goldilocks-core#177): spin_polarized=True must always come
    with a non-empty, non-zero starting_magnetization -- never implicit
    zero moments that relax to the non-magnetic solution."""
    state = magnetic_config(_IRON, is_magnetic(_IRON))

    assert state.ok
    assert state.value.spin_polarized is True
    assert state.value.magnetic_elements == ("Fe",)
    assert state.value.starting_magnetization == {"Fe": 0.5}


def test_non_magnetic_structure_has_no_starting_magnetization() -> None:
    state = magnetic_config(_SILICON, is_magnetic(_SILICON))

    assert state.value.spin_polarized is False
    assert state.value.starting_magnetization is None
    assert state.value.magnetic_elements == ()


def test_unavailable_magnetism_defaults_to_non_magnetic_not_blocked() -> None:
    """Unavailable (the heuristic tried and could not tell) is not the same
    as Blocked (nothing upstream could even attempt an answer) -- there is
    still a safe, documented default to fall back on here."""
    state = magnetic_config(_IRON, Unavailable(reason="ambiguous oxidation state"))

    assert state.ok
    assert state.value.spin_polarized is False
    assert any("could not be determined" in warning for warning in state.value.warnings)


def test_blocked_magnetism_propagates_as_blocked() -> None:
    state = magnetic_config(_IRON, Blocked(by="synthetic failure"))

    assert not state.ok
    assert state.status == "blocked"
    assert state.root_cause() == "synthetic failure"


def test_human_can_force_spin_polarization_on() -> None:
    state = magnetic_config(
        _SILICON,
        is_magnetic(_SILICON),
        human=MagneticConfigHumanInput(spin_polarized=True),
    )

    assert state.value.spin_polarized is True
    assert state.source == "human"


def test_giving_both_tot_and_starting_magnetization_does_not_error() -> None:
    """goldilocks-qe-pw-parameter-audit.md P0 finding #2: QE's own source
    (input.f90) does not error on this combination -- the design doc had
    briefly asserted it was a hard mutual exclusion elsewhere in itself and
    self-corrected. Both pass through unchanged."""
    state = magnetic_config(
        _IRON,
        is_magnetic(_IRON),
        human=MagneticConfigHumanInput(
            tot_magnetization=2.0, starting_magnetization={"Fe": 0.7}
        ),
    )

    assert state.ok
    assert state.value.tot_magnetization == 2.0
    assert state.value.starting_magnetization == {"Fe": 0.7}


def test_soc_enabled_on_a_magnetic_structure_keeps_the_magnetism() -> None:
    """A2b: enabling SOC must not silently discard magnetism -- both
    starting_magnetization and the angle1/angle2 direction QE's
    noncollinear mode needs must be present together."""
    state = magnetic_config(
        _IRON,
        is_magnetic(_IRON),
        human=MagneticConfigHumanInput(spin_orbit_coupling=True),
    )

    assert state.value.spin_polarized is True
    assert state.value.spin_orbit_enabled is True
    assert state.value.starting_magnetization == {"Fe": 0.5}
    assert state.value.angle1 == {"Fe": 0.0}
    assert state.value.angle2 == {"Fe": 0.0}


def test_soc_enabled_on_a_non_magnetic_structure_has_no_angles() -> None:
    """SOC without magnetism is only correct when the structure genuinely
    is non-magnetic -- confirmed by there being no is_magnetic override
    forcing it on here."""
    state = magnetic_config(
        _SILICON,
        is_magnetic(_SILICON),
        human=MagneticConfigHumanInput(spin_orbit_coupling=True),
    )

    assert state.value.spin_polarized is False
    assert state.value.spin_orbit_enabled is True
    assert state.value.angle1 is None
    assert state.value.angle2 is None


def test_soc_is_never_auto_enabled_even_when_needs_soc_says_yes() -> None:
    always_true = Resolved(True, is_magnetic(_IRON).provenance)

    state = magnetic_config(_IRON, is_magnetic(_IRON), needs_soc=always_true)

    assert state.value.spin_orbit_enabled is False
    assert any("consider enabling" in warning for warning in state.value.warnings)


def test_relabeled_structure_is_the_identity_for_now() -> None:
    """AFM species-splitting itself is a later extension -- the shape exists
    now, the behaviour ships FM-only."""
    state = magnetic_config(_IRON, is_magnetic(_IRON))

    assert state.value.relabeled_structure == _IRON
