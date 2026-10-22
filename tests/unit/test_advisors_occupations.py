from __future__ import annotations

import pytest
from pymatgen.core import Lattice, Structure
from support import magnetic_decision

from goldilocks_core.advisors.electron_count import ElectronCountDecision
from goldilocks_core.advisors.magnetic_config import (
    MagneticConfigFacts,
    MagneticConfigHumanInput,
    OrderingRecord,
)
from goldilocks_core.advisors.occupations import (
    OccupationsHumanInput,
    OccupationsLlmInput,
    occupations,
    whole_spin_channels,
)
from goldilocks_core.analysis.is_magnetic import is_magnetic
from goldilocks_core.analysis.is_metal import is_metal
from goldilocks_core.resolution import Blocked, Provenance, Resolved, Unavailable

_IRON = Structure(Lattice.cubic(2.87), ["Fe"], [[0.0, 0.0, 0.0]])
_SILICON = Structure(Lattice.cubic(5.43), ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])
_BCC_IRON = Structure(Lattice.cubic(2.87), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
_NON_METAL = Resolved("non_metal", Provenance(source="heuristic"))


def _electron_count(nelec: float):
    return Resolved(
        ElectronCountDecision(nelec=nelec, tot_charge=0.0),
        Provenance(source="heuristic"),
    )


def _applied_ordering(*, compensated: bool):
    """A collinear run on an applied antiferromagnetic ordering, with or
    without an operation that reverses every spin (``compensated``)."""
    facts = MagneticConfigFacts(
        cell=_BCC_IRON,  # not read by occupations
        ordering=OrderingRecord(
            requested="afm",
            applied="afm-1",
            compensated=compensated,
            criterion="smallest_cell",
        ),
        spin_polarized=True,
        magnetic_elements=("Fe",),
        starting_magnetization={"Fe1": 0.3, "Fe2": -0.3},
        tot_magnetization=None,
        spin_orbit_enabled=False,
        angle1=None,
        angle2=None,
    )
    return Resolved(facts, Provenance(source="human"))


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
        human=OccupationsHumanInput(smearing_type="mp", degauss=0.05),
    )

    assert state.ok
    assert state.value.occupations == "smearing"
    assert state.value.smearing_type == "mp"
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


def test_explicit_fixed_occupations_without_a_moment_keep_fixed_and_warn() -> None:
    """#138: an explicit fixed on a collinear run with no moment to pin
    stays fixed -- the gate refuses it -- and says why."""
    magnetic = magnetic_decision(_BCC_IRON, is_magnetic(_BCC_IRON))

    state = occupations(
        _NON_METAL,
        magnetic,
        _electron_count(32.0),
        human=OccupationsHumanInput(occupations="fixed"),
    )

    assert state.value.occupations == "fixed"
    assert state.value.tot_magnetization is None
    assert [warning.code for warning in state.value.warnings] == [
        "occupations.fixed_spin_polarized_needs_integer_tot_magnetization"
    ]


@pytest.mark.parametrize(
    "human",
    [
        # noncollinear: pw.x takes no tot_magnetization there (#146)
        MagneticConfigHumanInput(spin_orbit_coupling=True),
        # collinear, with the moment the warning asks for
        MagneticConfigHumanInput(tot_magnetization=2.0),
    ],
    ids=["spin-orbit", "tot_magnetization"],
)
def test_fixed_occupations_warn_only_for_a_collinear_run_without_a_moment(
    human: MagneticConfigHumanInput,
) -> None:
    magnetic = magnetic_decision(_BCC_IRON, is_magnetic(_BCC_IRON), human=human)

    state = occupations(Resolved("non_metal", Provenance(source="heuristic")), magnetic)

    assert state.value.occupations == "fixed"
    assert state.value.warnings == ()


def test_fixed_occupations_on_a_non_magnetic_system_has_no_warning() -> None:
    magnetic = magnetic_decision(_SILICON, is_magnetic(_SILICON))

    state = occupations(Resolved("non_metal", Provenance(source="heuristic")), magnetic)

    assert state.value.warnings == ()


def test_smearing_never_gets_the_fixed_spin_warning() -> None:
    magnetic = magnetic_decision(_IRON, is_magnetic(_IRON))

    state = occupations(is_metal(_IRON), magnetic)

    assert state.value.occupations == "smearing"
    assert state.value.warnings == ()


def test_a_spin_polarized_non_metal_without_a_moment_gets_smearing() -> None:
    """#138: LiFePO4, NiO and the other magnetic insulators got fixed
    occupations on nspin=2 without the integer tot_magnetization pw.x
    needs for that (QE 7.3 PW/src/input.f90:764-778), so the gate refused
    the default. Without a known moment the default is smearing."""
    magnetic = magnetic_decision(_BCC_IRON, is_magnetic(_BCC_IRON))

    state = occupations(_NON_METAL, magnetic, _electron_count(32.0))

    assert magnetic.value.spin_mode == "collinear"
    assert state.ok
    assert state.source == "heuristic"
    assert state.value.occupations == "smearing"
    assert state.value.smearing_type == "cold"
    assert state.value.degauss == 0.01
    assert state.value.tot_magnetization is None
    assert state.value.warnings == ()


def test_a_compensated_afm_non_metal_gets_fixed_with_a_zero_moment() -> None:
    """#138: an applied ordering with a spin-reversing symmetry has no net
    moment by construction, so fixed occupations pin tot_magnetization 0."""
    state = occupations(
        _NON_METAL, _applied_ordering(compensated=True), _electron_count(48.0)
    )

    assert state.value.occupations == "fixed"
    assert state.value.tot_magnetization == 0.0
    assert state.value.smearing_type is None
    assert state.value.warnings == ()


@pytest.mark.parametrize(
    ("compensated", "nelec"),
    [(False, 48.0), (True, 49.0)],
    ids=["no-spin-flip-symmetry", "odd-electron-count"],
)
def test_no_zero_moment_is_derived_without_symmetry_or_whole_channels(
    compensated: bool, nelec: float
) -> None:
    """A ferrimagnet's moment is not zero by symmetry, whatever its spin
    counts, and a zero moment cannot split an odd count into whole spin
    channels: neither gets a derived moment, so both get smearing."""
    state = occupations(
        _NON_METAL, _applied_ordering(compensated=compensated), _electron_count(nelec)
    )

    assert state.value.occupations == "smearing"
    assert state.value.tot_magnetization is None


@pytest.mark.parametrize(
    ("nelec", "moment", "whole"),
    [
        (32.0, 4.0, True),
        (33.0, 3.0, True),
        (32.0, 3.0, False),
        (8.0, 0.0, True),
        (9.0, 0.0, False),
        (8.0 + 1e-9, 0.0, True),  # summation noise
        (3.5, 0.5, False),  # nelup 2 is whole, neldw 1.5 is not
    ],
)
def test_whole_spin_channels_needs_both_channel_counts_whole(
    nelec: float, moment: float, whole: bool
) -> None:
    assert whole_spin_channels(nelec, moment) is whole


@pytest.mark.parametrize(
    ("moment", "nelec", "expected"),
    [
        (4.0, 32.0, "fixed"),
        (3.0, 33.0, "fixed"),
        (3.0, 32.0, "smearing"),  # nelup = 17.5: no whole spin channels
        (2.5, 32.0, "smearing"),  # not an integer
    ],
)
def test_a_human_moment_decides_fixed_only_when_both_spin_channels_are_whole(
    moment: float, nelec: float, expected: str
) -> None:
    """Fixed occupations fill nelup = (nelec + tot)/2 and neldw =
    (nelec - tot)/2 bands, both whole numbers only for an integer moment of
    the electron count's parity (#70). Any other moment still reaches the
    input, on smearing."""
    magnetic = magnetic_decision(
        _BCC_IRON,
        is_magnetic(_BCC_IRON),
        human=MagneticConfigHumanInput(tot_magnetization=moment),
    )

    state = occupations(_NON_METAL, magnetic, _electron_count(nelec))

    assert state.value.occupations == expected
    assert state.value.tot_magnetization == moment
    assert state.value.warnings == ()


@pytest.mark.parametrize(("nelec", "expected"), [(9.0, "smearing"), (8.0, "fixed")])
@pytest.mark.parametrize(
    ("spin_orbit_coupling", "mode"), [(None, "none"), (True, "noncollinear")]
)
def test_an_odd_electron_count_without_spin_gets_smearing(
    nelec: float, expected: str, spin_orbit_coupling: bool | None, mode: str
) -> None:
    """With nspin=1 fixed occupations fill NINT(nelec)/2 doubly occupied
    bands (QE 7.3 PW/src/iweights.f90), which drops an odd electron, so
    the default never pairs them (#70). A spin-orbit run without moments
    fills single spinor bands, but keeps time reversal: every band has a
    degenerate Kramers partner, and an odd count half-fills a pair."""
    magnetic = magnetic_decision(
        _SILICON,
        is_magnetic(_SILICON),
        human=MagneticConfigHumanInput(spin_orbit_coupling=spin_orbit_coupling),
    )

    state = occupations(_NON_METAL, magnetic, _electron_count(nelec))

    assert magnetic.value.spin_mode == mode
    assert magnetic.value.spin_polarized is False
    assert state.value.occupations == expected
    assert state.value.tot_magnetization is None


def test_a_magnetic_spin_orbit_non_metal_gets_fixed_without_a_moment() -> None:
    """A spin-polarized noncollinear run takes no tot_magnetization, breaks
    time reversal and fills single-electron spinor bands (#146), so neither
    the moment nor the parity matters."""
    magnetic = magnetic_decision(
        _BCC_IRON,
        is_magnetic(_BCC_IRON),
        human=MagneticConfigHumanInput(spin_orbit_coupling=True),
    )

    state = occupations(_NON_METAL, magnetic, _electron_count(33.0))

    assert magnetic.value.spin_mode == "noncollinear"
    assert magnetic.value.spin_polarized is True
    assert state.value.occupations == "fixed"
    assert state.value.tot_magnetization is None
    assert state.value.warnings == ()


def test_explicit_fixed_on_a_compensated_cell_derives_the_zero_moment() -> None:
    state = occupations(
        _NON_METAL,
        _applied_ordering(compensated=True),
        _electron_count(48.0),
        human=OccupationsHumanInput(occupations="fixed"),
    )

    assert state.value.occupations == "fixed"
    assert state.value.tot_magnetization == 0.0
    assert state.value.warnings == ()


@pytest.mark.parametrize("choice", ["smearing", "tetrahedra_opt"])
def test_smearing_and_tetrahedra_carry_the_human_moment_never_the_derived_zero(
    choice: str,
) -> None:
    """The derived 0 holds only under fixed occupations: on smearing it
    would pin a moment nobody asked for. A human moment applies to every
    collinear run."""
    human = OccupationsHumanInput(occupations=choice)
    compensated = occupations(
        _NON_METAL, _applied_ordering(compensated=True), _electron_count(48.0), human
    )
    requested = occupations(
        _NON_METAL,
        magnetic_decision(
            _BCC_IRON,
            is_magnetic(_BCC_IRON),
            human=MagneticConfigHumanInput(tot_magnetization=2.0),
        ),
        _electron_count(32.0),
        human,
    )

    assert compensated.value.occupations == choice
    assert compensated.value.tot_magnetization is None
    assert requested.value.tot_magnetization == 2.0


def test_a_metal_carries_the_human_moment() -> None:
    magnetic = magnetic_decision(
        _BCC_IRON,
        is_magnetic(_BCC_IRON),
        human=MagneticConfigHumanInput(tot_magnetization=4.0),
    )

    state = occupations(is_metal(_BCC_IRON), magnetic, _electron_count(32.0))

    assert state.value.occupations == "smearing"
    assert state.value.tot_magnetization == 4.0


@pytest.mark.parametrize(
    "electron_count",
    [
        Unavailable(reason="pseudopotential table not installed"),
        Blocked(by="pseudopotential selection failed"),
    ],
    ids=["unavailable", "blocked"],
)
def test_an_unknown_electron_count_never_blocks_occupations(electron_count) -> None:
    """A table not installed yet is a normal preview, and nbnd and the
    writer already block on the count: occupations resolves without the
    parity test. The derived 0 needs a known even count; a human integer
    moment does not."""
    unpolarized = occupations(
        _NON_METAL, magnetic_decision(_SILICON, is_magnetic(_SILICON)), electron_count
    )
    compensated = occupations(
        _NON_METAL, _applied_ordering(compensated=True), electron_count
    )
    requested = occupations(
        _NON_METAL,
        magnetic_decision(
            _BCC_IRON,
            is_magnetic(_BCC_IRON),
            human=MagneticConfigHumanInput(tot_magnetization=4.0),
        ),
        electron_count,
    )

    assert unpolarized.ok
    assert unpolarized.value.occupations == "fixed"
    assert compensated.value.occupations == "smearing"
    assert compensated.value.tot_magnetization is None
    assert requested.value.occupations == "fixed"
    assert requested.value.tot_magnetization == 4.0
