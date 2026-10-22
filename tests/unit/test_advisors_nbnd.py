from __future__ import annotations

import pytest
from pymatgen.core import Lattice, Structure
from support import magnetic_decision

from goldilocks_core.advisors.magnetic_config import (
    MagneticConfigFacts,
    MagneticConfigHumanInput,
    SpinMode,
)
from goldilocks_core.advisors.nbnd import NbndHumanInput, NbndLlmInput, nbnd
from goldilocks_core.advisors.occupations import OccupationsDecision
from goldilocks_core.analysis.is_magnetic import is_magnetic
from goldilocks_core.resolution import Blocked, FieldState, Provenance, Resolved

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

    assert state.value.nbnd == 11  # nelup = INT(22)/2 = 11, neldw = 10


def test_metal_gets_twenty_percent_more_when_that_exceeds_the_floor() -> None:
    # nelec=200 -> nelup = neldw = 100; NINT(1.2*100)=120 beats 100+4=104.
    state = nbnd(200.0, _SMEARING)

    assert state.value.nbnd == 120


def test_metal_gets_at_least_four_more_when_twenty_percent_is_smaller() -> None:
    # nelec=10 -> nelup = neldw = 5; NINT(1.2*5)=6, but 5+4=9 is the floor.
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


def test_extra_bands_driving_the_total_non_positive_is_unavailable() -> None:
    """Regression for #35 (v2 epic 9, #9): extra_bands is a legitimately
    -signed adjustment (can't just be Field(gt=0)), but a sufficiently
    negative value used to silently drive the combined nbnd to zero or
    negative, reported as a resolved success rather than degrading."""
    state = nbnd(20.0, _FIXED, human=NbndHumanInput(extra_bands=-15))

    assert not state.ok
    assert state.status == "unavailable"


def test_llm_extra_bands_used_when_no_human_override() -> None:
    state = nbnd(20.0, _FIXED, llm=NbndLlmInput(extra_bands=3))

    assert state.value.nbnd == 13
    assert state.source == "llm"


def test_spin_polarized_system_gets_an_informational_note() -> None:
    magnetic = magnetic_decision(_IRON, is_magnetic(_IRON))

    state = nbnd(16.0, _FIXED, magnetic=magnetic)

    assert any(
        "not the number of bands" in warning.message for warning in state.value.warnings
    )


def test_non_magnetic_system_has_no_spin_note() -> None:
    magnetic = magnetic_decision(_SILICON, is_magnetic(_SILICON))

    state = nbnd(8.0, _FIXED, magnetic=magnetic)

    assert state.value.warnings == ()


def test_noncollinear_insulator_uses_the_full_electron_count_not_half() -> None:
    """QE's noncollinear bands hold one electron each (a two-component
    spinor), not two -- twice the band count of a comparable collinear
    run (v2 epic 9, #9)."""
    magnetic = magnetic_decision(
        _SILICON,
        is_magnetic(_SILICON),
        human=MagneticConfigHumanInput(spin_orbit_coupling=True),
    )

    state = nbnd(20.0, _FIXED, magnetic=magnetic)

    assert state.value.nbnd == 20


def test_noncollinear_metal_doubles_the_padded_per_spin_count() -> None:
    # nelec=200 -> nelup = neldw = 100, padded to MAX(120, 120, 120, 104)
    # = 120, then doubled (setup.f90 pads before doubling, #162).
    magnetic = magnetic_decision(
        _SILICON,
        is_magnetic(_SILICON),
        human=MagneticConfigHumanInput(spin_orbit_coupling=True),
    )

    state = nbnd(200.0, _SMEARING, magnetic=magnetic)

    assert state.value.nbnd == 240


def test_noncollinear_gets_its_own_note_not_the_nspin_two_one() -> None:
    magnetic = magnetic_decision(
        _IRON,
        is_magnetic(_IRON),
        human=MagneticConfigHumanInput(spin_orbit_coupling=True),
    )

    state = nbnd(16.0, _FIXED, magnetic=magnetic)

    assert any("one electron" in warning.message for warning in state.value.warnings)
    assert not any(
        "not the number of bands" in warning.message for warning in state.value.warnings
    )


# #162: nbnd is pw.x's own default, QE 7.3 PW/src/setup.f90:326-345, with
# nelup/neldw from Modules/electrons_base.f90:376-428 (set_nelup_neldw):
#   base = MAX(NINT(nelec/2), NINT(nelup), NINT(neldw))
#   smearing/tetrahedra: MAX(NINT(1.2*nelec/2), NINT(1.2*nelup),
#                            NINT(1.2*neldw), base + 4)
#   noncolin: doubled after the padding.
# Without tot_magnetization nelup = INT(nelec+1)/2 and neldw = nelec - nelup;
# with an integer one, nelup/neldw = (INT(nelec) +/- tot)/2. NINT rounds
# half away from zero (Python's round() rounds half to even).
_SI_PRIMITIVE = Structure(
    Lattice([[0.0, 2.715, 2.715], [2.715, 0.0, 2.715], [2.715, 2.715, 0.0]]),
    ["Si", "Si"],
    [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
)
_FEPT = Structure(
    Lattice.tetragonal(2.73, 3.71), ["Fe", "Pt"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
)


def _spin_facts(mode: SpinMode) -> FieldState[MagneticConfigFacts]:
    """A resolved magnetic decision of spin mode ``mode``: diamond Si for
    ``none``, L1_0 FePt otherwise (spin-orbit coupling on for
    ``noncollinear``)."""
    if mode == "none":
        facts = magnetic_decision(_SI_PRIMITIVE, is_magnetic(_SI_PRIMITIVE))
    else:
        human = MagneticConfigHumanInput(spin_orbit_coupling=mode == "noncollinear")
        facts = magnetic_decision(_FEPT, is_magnetic(_FEPT), human=human)
    assert facts.value.spin_mode == mode
    return facts


def _occupations(
    scheme: str, tot_magnetization: float | None = None
) -> FieldState[OccupationsDecision]:
    if scheme == "fixed":
        decision = OccupationsDecision(
            occupations="fixed",
            smearing_type=None,
            degauss=None,
            tot_magnetization=tot_magnetization,
        )
    else:
        decision = OccupationsDecision(
            occupations="smearing",
            smearing_type="cold",
            degauss=0.01,
            tot_magnetization=tot_magnetization,
        )
    return Resolved(decision, Provenance(source="heuristic"))


@pytest.mark.parametrize(
    ("nelec", "scheme", "mode", "tot_magnetization", "expected"),
    [
        # Spin-orbit smearing: padded per spin channel, then doubled.
        # Pb: nelup = neldw = 7; 2 * MAX(NINT(8.4)=8, 8, 8, 7+4) = 22.
        pytest.param(14.0, "smearing", "noncollinear", None, 22, id="Pb-soc-14"),
        # Bi: nelup = neldw = 15; 2 * MAX(18, 18, 18, 15+4) = 38.
        pytest.param(30.0, "smearing", "noncollinear", None, 38, id="Bi-soc-30"),
        # FePt: nelup = neldw = 17; 2 * MAX(NINT(20.4)=20, 20, 20, 17+4) = 42.
        pytest.param(34.0, "smearing", "noncollinear", None, 42, id="FePt-soc-34"),
        # nelup = 7, neldw = 6; base MAX(NINT(6.5)=7, 7, 6) = 7;
        # 2 * MAX(NINT(7.8)=8, NINT(8.4)=8, NINT(7.2)=7, 7+4) = 22: even.
        pytest.param(13.0, "smearing", "noncollinear", None, 22, id="soc-odd-13"),
        # Collinear smearing. MoS2: nelup = neldw = 26;
        # MAX(NINT(31.2)=31, 31, 31, 26+4) = 31.
        pytest.param(52.0, "smearing", "none", None, 31, id="MoS2-52"),
        # nelup = 23, neldw = 22; base MAX(NINT(22.5)=23, 23, 22) = 23;
        # MAX(NINT(27.0)=27, NINT(27.6)=28, NINT(26.4)=26, 23+4) = 28.
        pytest.param(45.0, "smearing", "none", None, 28, id="nelup-term-45"),
        # nelup = neldw = 5; MAX(6, 6, 6, 5+4) = 9.
        pytest.param(10.0, "smearing", "none", None, 9, id="floor-10"),
        # Fixed: nelup = INT(26)/2 = 13, neldw = 12; MAX(NINT(12.5)=13, 13, 12).
        pytest.param(25.0, "fixed", "none", None, 13, id="fixed-25"),
        # nelup = 11, neldw = 10; MAX(NINT(10.5)=11, 11, 10) = 11.
        pytest.param(21.0, "fixed", "none", None, 11, id="fixed-21"),
        # Spin-orbit fixed: 2 * MAX(10, 10, 10) = 20.
        pytest.param(20.0, "fixed", "noncollinear", None, 20, id="soc-fixed-20"),
        # nspin=2 fixed: nelup = (192+16)/2 = 104, neldw = 88; MAX(96, 104, 88).
        pytest.param(192.0, "fixed", "collinear", 16.0, 104, id="LiFePO4-fixed"),
        # nelup = 7, neldw = 5; MAX(6, 7, 5) = 7.
        pytest.param(12.0, "fixed", "collinear", 2.0, 7, id="O2-fixed"),
        # nelup = 6.5, neldw = 5.5; MAX(6, NINT(6.5)=7, NINT(5.5)=6) = 7,
        # where round() gives MAX(6, 6, 6) = 6.
        pytest.param(12.0, "fixed", "collinear", 1.0, 7, id="nint-half-up-fixed"),
        # Same spin channels under smearing: base 7; MAX(NINT(7.2)=7,
        # NINT(7.8)=8, NINT(6.6)=7, 7+4) = 11, where round() gives 10.
        pytest.param(12.0, "smearing", "collinear", 1.0, 11, id="nint-half-up"),
        # nspin=2 smearing: nelup = 25, neldw = 11; base MAX(18, 25, 11) = 25;
        # MAX(NINT(21.6)=22, NINT(30.0)=30, NINT(13.2)=13, 25+4) = 30.
        pytest.param(36.0, "smearing", "collinear", 14.0, 30, id="collinear-tot-14"),
        # A negative moment makes the minority channel the larger one:
        # nelup = 11, neldw = 25; MAX(NINT(21.6)=22, NINT(13.2)=13,
        # NINT(30.0)=30, 25+4) = 30.
        pytest.param(36.0, "smearing", "collinear", -14.0, 30, id="collinear-tot--14"),
        # nelup = 5, neldw = 7; MAX(6, 5, 7) = 7.
        pytest.param(12.0, "fixed", "collinear", -2.0, 7, id="fixed-tot--2"),
        # Without tot_magnetization nspin=2 sizes like nspin=1:
        # nelup = 7, neldw = 6, as soc-odd-13 without the doubling.
        pytest.param(13.0, "smearing", "collinear", None, 11, id="collinear-13"),
    ],
)
def test_nbnd_is_pw_x_own_default(
    nelec: float,
    scheme: str,
    mode: SpinMode,
    tot_magnetization: float | None,
    expected: int,
) -> None:
    state = nbnd(
        nelec, _occupations(scheme, tot_magnetization), magnetic=_spin_facts(mode)
    )

    assert state.value.nbnd == expected
    assert state.source == "heuristic"


def test_the_moment_counts_only_where_the_writer_renders_it() -> None:
    """``tot_magnetization`` is written for a collinear spin-polarized run
    only (nspin=2; pw.x refuses it otherwise), so nbnd sizes from it only
    there: the same decision on a run without spin polarization sizes as
    nelup = neldw = 96."""
    occupations = _occupations("fixed", 16.0)

    unpolarized = nbnd(192.0, occupations, magnetic=_spin_facts("none"))
    collinear = nbnd(192.0, occupations, magnetic=_spin_facts("collinear"))

    assert (unpolarized.value.nbnd, collinear.value.nbnd) == (96, 104)
