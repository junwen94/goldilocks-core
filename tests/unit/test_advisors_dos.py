from __future__ import annotations

from goldilocks_core.advisors.dos import (
    DEFAULT_DELTA_E,
    DEFAULT_NGAUSS,
    DosHumanInput,
    DosLlmInput,
    dos_settings,
)
from goldilocks_core.advisors.occupations import OccupationsDecision
from goldilocks_core.resolution import Blocked, Provenance, Resolved

_SMEARING = Resolved(
    OccupationsDecision(occupations="smearing", smearing_type="cold", degauss=0.02),
    Provenance(source="heuristic"),
)
_TETRAHEDRA = Resolved(
    OccupationsDecision(occupations="tetrahedra_opt", smearing_type=None, degauss=None),
    Provenance(source="heuristic"),
)


def test_defaults_match_qe_and_aiida_documented_values() -> None:
    state = dos_settings(_TETRAHEDRA)

    assert state.ok
    assert state.value.delta_e == DEFAULT_DELTA_E
    assert state.value.ngauss == DEFAULT_NGAUSS
    assert state.value.emin is None
    assert state.value.emax is None
    assert state.source == "heuristic"


def test_broadening_inherits_the_nscf_steps_own_smearing_width() -> None:
    state = dos_settings(_SMEARING)

    assert state.value.broadening == 0.02


def test_broadening_stays_none_when_nscf_used_a_tetrahedron_method() -> None:
    """No smearing width exists for a tetrahedron-method nscf step --
    dos.x's own bz_sum default already switches to tetrahedron
    integration whenever degauss is unset, so leaving it None here is
    correct, not a gap."""
    state = dos_settings(_TETRAHEDRA)

    assert state.value.broadening is None


def test_human_overrides_every_field() -> None:
    state = dos_settings(
        _SMEARING,
        human=DosHumanInput(
            emin=-5.0, emax=5.0, delta_e=0.005, broadening=0.01, ngauss=1
        ),
    )

    assert state.value.emin == -5.0
    assert state.value.emax == 5.0
    assert state.value.delta_e == 0.005
    assert state.value.broadening == 0.01
    assert state.value.ngauss == 1
    assert state.source == "human"


def test_llm_delta_e_used_when_no_human_override() -> None:
    state = dos_settings(_SMEARING, llm=DosLlmInput(delta_e=0.02))

    assert state.value.delta_e == 0.02
    assert state.source == "llm"


def test_blocked_nscf_occupations_propagates_as_blocked() -> None:
    state = dos_settings(Blocked(by="synthetic failure"))

    assert not state.ok
    assert state.root_cause() == "synthetic failure"
