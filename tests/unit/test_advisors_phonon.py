"""The phonon tasks' ph.x, q2r.x and matdyn.x settings and the notices of a
phonon run (#215): defaults and request values, the bounds ph.x checks, each
notice on its own condition, the acoustic-sum-rule fallback against one
task's memory, loto_2d, and the phonon DOS mesh."""

from __future__ import annotations

import dataclasses

import pytest
from pydantic import ValidationError
from pymatgen.core import Structure
from support import (
    INTERNAL_TEXT,
    framed_magnetic_cell,
    geometry_facts,
    graphene_monolayer,
    mc3d_42058_silicon,
    rutile_titanium_dioxide,
    written_afm_nickel_oxide,
    written_boron_nitride_monolayer,
    written_cell,
    written_magnetic_cell,
)

from goldilocks_core.advisors.boundary import BoundaryFacts
from goldilocks_core.advisors.hubbard_u import HubbardU, HubbardUDecision
from goldilocks_core.advisors.job_resources import JobDecision
from goldilocks_core.advisors.occupations import OccupationsDecision
from goldilocks_core.advisors.phonon import (
    LOOSE_FORCE_THRESHOLD_WARNING,
    PHONON_CITATIONS,
    WARNING_CATALOGUE,
    PhononDosHumanInput,
    PhononHumanInput,
    Q2rHumanInput,
    phonon_dos_settings,
    phonon_notices,
    phonon_settings,
    q2r_settings,
)
from goldilocks_core.advisors.q_sampling import QSamplingDecision
from goldilocks_core.inputs.hpc import Hardware, HpcProfile, Partition, load_hpc_profile
from goldilocks_core.resolution import (
    Blocked,
    Provenance,
    Resolved,
    Unavailable,
    Warning,
)


def _resolved(value: object, source: str = "heuristic") -> Resolved[object]:
    return Resolved(value, Provenance(source=source))


def _occupations(occupations: str = "fixed") -> Resolved[OccupationsDecision]:
    smearing = occupations == "smearing"
    decision = OccupationsDecision(
        occupations=occupations,
        smearing_type="cold" if smearing else None,
        degauss=0.01 if smearing else None,
    )
    return Resolved(
        decision, Provenance.from_field_sources({"occupations": "heuristic"})
    )


def _notices(
    first_step: str = "relax",
    *,
    occupations: Resolved[OccupationsDecision] | None = None,
    structure: Structure | None = None,
    vacuum_axes: tuple[str, ...] = (),
    isolation: str = "periodic",
    spin_polarized: bool = False,
    spin_orbit: bool = False,
    hubbard: HubbardUDecision | None = None,
    d3: bool = False,
) -> tuple[Warning, ...]:
    """The notices of a relax-phonon run of rutile TiO2, bulk, periodic,
    non-magnetic, without +U or D3 and on fixed occupations, which raises
    none, with one condition changed."""
    magnetic = written_cell(structure or rutile_titanium_dioxide())
    magnetic = _resolved(
        dataclasses.replace(
            magnetic.value,
            spin_polarized=spin_polarized,
            spin_orbit_enabled=spin_orbit,
        )
    )
    dimensionality = "2d" if vacuum_axes else "3d"
    return phonon_notices(
        first_step,
        occupations or _occupations(),
        _resolved(geometry_facts(dimensionality, vacuum_axes)),
        _resolved(BoundaryFacts(isolation)),
        magnetic,
        _resolved(hubbard or HubbardUDecision()),
        d3=d3,
    )


def _codes(notices: tuple[Warning, ...]) -> list[str]:
    return [notice.code for notice in notices]


_NICKEL_OXIDE_WITH_U = {
    "structure": written_afm_nickel_oxide(),
    "spin_polarized": True,
    "hubbard": HubbardUDecision(
        u_values=(HubbardU("Ni", "3d", 6.2, "table"),), projectors="atomic"
    ),
}
"""AFM NiO, collinear, with the table's U on Ni and atomic projectors."""


def test_the_ph_x_defaults_are_the_balanced_protocol_s_with_its_note() -> None:
    """aiida-quantumespresso's balanced ph protocol: Quantum ESPRESSO's own
    defaults are 1e-12, 0.7 and 4."""
    state = phonon_settings()

    assert state.ok
    decision = state.value
    assert (decision.tr2_ph, decision.alpha_mix, decision.nmix_ph) == (1e-18, 0.4, 8)
    assert decision.warnings == ()
    assert state.source == "heuristic"
    assert state.provenance.field_sources == {
        "tr2_ph": "heuristic",
        "alpha_mix": "heuristic",
        "nmix_ph": "heuristic",
    }
    assert "aiida-quantumespresso's ph protocol" in state.provenance.source_note
    assert state.provenance.citations == PHONON_CITATIONS
    assert PHONON_CITATIONS == (
        "S. P. Huber et al., npj Comput. Mater. 7, 136 (2021)",
        "S. Baroni et al., Rev. Mod. Phys. 73, 515 (2001)",
    )


def test_request_values_are_used_as_given_and_read_human() -> None:
    state = phonon_settings(PhononHumanInput(tr2_ph=1e-16, alpha_mix=0.7, nmix_ph=4))

    decision = state.value
    assert (decision.tr2_ph, decision.alpha_mix, decision.nmix_ph) == (1e-16, 0.7, 4)
    assert state.source == "human"
    assert set(state.provenance.field_sources.values()) == {"human"}
    assert state.provenance.source_note is None
    assert state.provenance.citations == ()


def test_one_request_value_leaves_the_others_at_the_protocol() -> None:
    state = phonon_settings(PhononHumanInput(alpha_mix=0.3))

    assert (state.value.tr2_ph, state.value.alpha_mix, state.value.nmix_ph) == (
        1e-18,
        0.3,
        8,
    )
    assert state.provenance.field_sources == {
        "tr2_ph": "heuristic",
        "alpha_mix": "human",
        "nmix_ph": "heuristic",
    }
    assert state.source == "human"
    assert state.provenance.citations == PHONON_CITATIONS


def test_the_notices_are_the_phonon_record_s_warnings() -> None:
    notices = _notices("scf", d3=True)

    assert phonon_settings(notices=notices).value.warnings == notices


@pytest.mark.parametrize(
    ("bag", "setting"),
    [
        (PhononHumanInput, {"tr2_ph": 0.0}),
        (PhononHumanInput, {"alpha_mix": 1.5}),
        (PhononHumanInput, {"alpha_mix": -0.1}),
        (PhononHumanInput, {"nmix_ph": 0}),
        (Q2rHumanInput, {"asr": "all"}),
        (PhononDosHumanInput, {"dos_q_distance": 0.0}),
    ],
)
def test_a_value_outside_a_setting_s_bounds_is_refused(bag, setting) -> None:
    """ph.x stops unless tr2_ph > 0 and 0 <= alpha_mix <= 1 (QE 7.3
    phq_readin.f90:409-416); nmix_ph mixes at least one earlier iteration,
    and of matdyn.x's acoustic sum rules these tasks offer crystal and
    simple."""
    with pytest.raises(ValidationError):
        bag(**setting)


def test_the_bounds_themselves_are_accepted() -> None:
    bag = PhononHumanInput(alpha_mix=1.0, nmix_ph=1)

    assert (bag.alpha_mix, bag.nmix_ph) == (1.0, 1)
    assert PhononHumanInput(alpha_mix=0.0).alpha_mix == 0.0


def test_a_fixed_relaxation_of_a_bulk_oxide_raises_no_notice() -> None:
    assert _notices() == ()


@pytest.mark.parametrize(
    ("first_step", "codes"),
    [
        ("scf", ["phonon.relax_first"]),
        ("relax", []),
        ("vc-relax", ["phonon.vc_relax_input_cell"]),
    ],
)
def test_the_first_step_decides_the_relaxation_notices(
    first_step: str, codes: list[str]
) -> None:
    """The phonon task computes phonons at the written geometry; the
    vc-relax-phonon task sizes its meshes on the input cell."""
    assert _codes(_notices(first_step)) == codes


def test_the_relaxation_notice_asks_for_the_same_projectors() -> None:
    (notice,) = _notices("scf")

    assert "hubbard_projectors" in notice.message
    assert "relax-phonon" in notice.message
    assert notice.level == "info"


def test_a_fixed_periodic_slab_gets_the_2d_lo_to_warning() -> None:
    """Its non-analytic term would be the 3D one, which depends on the
    vacuum width."""
    notices = _notices(structure=written_boron_nitride_monolayer(), vacuum_axes=("c",))

    assert _codes(notices) == ["phonon.slab_lo_to_needs_2d_truncation"]
    assert "isolation=slab" in notices[0].message


@pytest.mark.parametrize(
    "case",
    [{"isolation": "slab"}, {"occupations": _occupations("smearing")}],
    ids=["2d-truncation", "smearing"],
)
def test_no_lo_to_warning_with_2d_truncation_or_without_fixed_occupations(
    case,
) -> None:
    notices = _notices(
        structure=written_boron_nitride_monolayer(), vacuum_axes=("c",), **case
    )

    assert notices == ()


@pytest.mark.parametrize(
    ("spin_polarized", "codes"),
    [(True, ["phonon.magnetic_noncollinear_experimental"]), (False, [])],
    ids=["magnetic", "non-magnetic"],
)
def test_only_magnetic_spin_orbit_runs_get_the_experimental_warning(
    spin_polarized: bool, codes: list[str]
) -> None:
    """Quantum ESPRESSO's PHonon examples call the magnetic spin-orbit case
    experimental; a non-magnetic one (fcc Pt) is not."""
    notices = _notices(spin_polarized=spin_polarized, spin_orbit=True)

    assert _codes(notices) == codes


def test_a_collinear_magnet_gets_no_spin_orbit_warning() -> None:
    assert _notices(spin_polarized=True) == ()


def test_a_hubbard_u_gets_the_dfpt_u_warning() -> None:
    notices = _notices(**_NICKEL_OXIDE_WITH_U)

    assert _codes(notices) == ["phonon.hubbard_experimental"]
    assert notices[0].level == "warning"


@pytest.mark.parametrize(("d3", "codes"), [(True, ["phonon.d3_two_body"]), (False, [])])
def test_only_a_d3_run_gets_the_two_body_notice(d3: bool, codes: list[str]) -> None:
    """d3hess.x's Hessian is two-body only, so the first step writes
    dftd3_threebody = .false. with D3."""
    notices = _notices(d3=d3)

    assert _codes(notices) == codes
    if d3:
        assert "dftd3_threebody = .false." in notices[0].message


def test_unresolved_facts_raise_no_notice_of_their_own() -> None:
    """A blocked record blocks the run on its own; the notices that read
    it are left out rather than guessed."""
    blocked = Blocked(by="pseudopotential table sssp is not installed")

    notices = phonon_notices(
        "scf", blocked, blocked, blocked, blocked, blocked, d3=False
    )

    assert _codes(notices) == ["phonon.relax_first"]


def test_the_catalogue_holds_every_notice_and_the_relax_warning() -> None:
    """The force-threshold warning is carried by the relax record."""
    by_code = {warning.code: warning for warning in WARNING_CATALOGUE}

    assert by_code["phonon.loose_force_threshold"] == LOOSE_FORCE_THRESHOLD_WARNING
    assert (
        LOOSE_FORCE_THRESHOLD_WARNING.level,
        LOOSE_FORCE_THRESHOLD_WARNING.category,
    ) == (
        "warning",
        "relax",
    )
    assert by_code["phonon.asr_simple_fallback"].category == "q2r"
    raised = [
        *_notices("scf", d3=True, **_NICKEL_OXIDE_WITH_U),
        *_notices("vc-relax"),
        *_notices(structure=written_boron_nitride_monolayer(), vacuum_axes=("c",)),
        *_notices(spin_polarized=True, spin_orbit=True),
    ]
    for notice in raised:
        catalogued = by_code[notice.code]
        assert (catalogued.level, catalogued.category) == (notice.level, "phonon")


_AFM_NICKEL_OXIDE_Q = _resolved(
    QSamplingDecision(mesh=(9, 9, 9), q_distance=0.3, n_irr_q=85, work=9477)
)
"""AFM NiO's default q-mesh (its q_sampling record)."""


def _job(partition: str = "scarf") -> Resolved[JobDecision]:
    return _resolved(
        JobDecision(
            partition=partition,
            nodes=1,
            ntasks=64,
            ntasks_per_node=64,
            walltime_h=168.0,
            max_seconds_by_step={},
            memory_per_process_mb=50.0,
            memory_total_mb=3200.0,
            account=None,
            timer_seconds=574560,
        )
    )


def _small_memory_profile() -> HpcProfile:
    """A node of 64 cores and 4 GB: 64 MB per task, below AFM NiO's
    asr='crystal' projection at 9x9x9."""
    hardware = Hardware(
        cores_per_node=64, mem_per_node_gb=4.0, max_nodes=10, max_walltime_h=168
    )
    return HpcProfile(
        name="small",
        scheduler="slurm",
        launcher="srun",
        modules={},
        partitions={"scarf": Partition("scarf", hardware, default=True)},
    )


def _q2r(
    isolation: str = "periodic",
    hpc: HpcProfile | None = None,
    human: Q2rHumanInput | None = None,
    *,
    q=_AFM_NICKEL_OXIDE_Q,
    job=None,
):
    return q2r_settings(
        written_magnetic_cell(written_afm_nickel_oxide()),
        q,
        _resolved(BoundaryFacts(isolation)),
        job or _job(),
        hpc or load_hpc_profile("scarf"),
        human,
    )


def test_asr_is_crystal_when_its_projection_fits_one_task_s_memory() -> None:
    """matdyn.x's projection holds (18 nat + 29) arrays of 9 nat^2 N_q
    doubles: 80.9 MB for AFM NiO's 4 atoms at 9x9x9, well within SCARF's
    4000 MB per task (250 GB over 64 cores)."""
    state = _q2r()

    assert state.ok
    assert state.value.asr == "crystal"
    assert state.value.asr_memory_mb == 80.89
    assert state.value.loto_2d is False
    assert state.value.warnings == ()
    assert state.provenance.field_sources == {"asr": "heuristic"}


def test_asr_falls_back_to_simple_beyond_one_task_s_memory() -> None:
    state = _q2r(hpc=_small_memory_profile())

    assert state.value.asr == "simple"
    assert state.value.asr_memory_mb == 80.89
    (notice,) = state.value.warnings
    assert notice.code == "phonon.asr_simple_fallback"
    assert (notice.level, notice.category) == ("info", "q2r")
    assert "80.9 MB" in notice.message
    assert "64 MB" in notice.message
    assert "scarf" in notice.message
    assert not INTERNAL_TEXT.findall(notice.message)
    assert state.provenance.field_sources == {"asr": "heuristic"}


@pytest.mark.parametrize(
    ("asr", "hpc"),
    [("crystal", _small_memory_profile), ("simple", lambda: load_hpc_profile("scarf"))],
)
def test_a_request_asr_is_used_as_given(asr: str, hpc) -> None:
    state = _q2r(hpc=hpc(), human=Q2rHumanInput(asr=asr))

    assert state.value.asr == asr
    assert state.value.warnings == ()
    assert state.provenance.field_sources == {"asr": "human"}
    assert state.source == "human"


@pytest.mark.parametrize(
    ("isolation", "loto_2d"),
    [("slab", True), ("periodic", False), ("isolated", False)],
)
def test_loto_2d_follows_the_slab_boundary(isolation: str, loto_2d: bool) -> None:
    """With QE's 2D Coulomb truncation q2r.x and matdyn.x take the 2D
    non-analytic term (rigid.f90:129-137)."""
    assert _q2r(isolation).value.loto_2d is loto_2d


@pytest.mark.parametrize(
    "inputs",
    [
        {"q": Blocked(by="pseudopotential table sssp is not installed")},
        {"job": Unavailable(reason="HPC profile scarf has no partition gpu2")},
    ],
    ids=["q", "job"],
)
def test_q2r_is_blocked_by_an_unresolved_q_mesh_or_job(inputs) -> None:
    state = _q2r(**inputs)

    assert isinstance(state, Blocked)


@pytest.mark.parametrize(
    ("cell", "mesh"),
    [
        (lambda: framed_magnetic_cell(mc3d_42058_silicon()), (21, 21, 21)),
        (lambda: framed_magnetic_cell(graphene_monolayer()), (30, 30, 1)),
    ],
    ids=["silicon", "graphene"],
)
def test_the_dos_mesh_is_0_10_per_angstrom_on_the_written_cell(cell, mesh) -> None:
    """The dos task's nscf spacing, through kmesh: one division along a
    vacuum axis."""
    state = phonon_dos_settings(cell())

    assert state.ok
    assert state.value.mesh == mesh
    assert (state.value.dos_q_distance, state.value.delta_e) == (0.1, 1.0)
    assert state.source == "heuristic"
    assert state.provenance.field_sources == {"dos_q_distance": "heuristic"}
    assert "0.10" in state.provenance.source_note


def test_a_request_dos_q_distance_is_used_as_given() -> None:
    state = phonon_dos_settings(
        framed_magnetic_cell(mc3d_42058_silicon()),
        PhononDosHumanInput(dos_q_distance=0.2),
    )

    assert state.value.mesh == (11, 11, 11)
    assert state.value.dos_q_distance == 0.2
    assert state.provenance.field_sources == {"dos_q_distance": "human"}
    assert state.provenance.source_note is None


def test_the_dos_mesh_is_blocked_by_an_unresolved_cell() -> None:
    state = phonon_dos_settings(Blocked(by="no compatible pseudopotential table"))

    assert isinstance(state, Blocked)
