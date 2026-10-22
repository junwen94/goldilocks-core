from __future__ import annotations

import pytest
from pymatgen.core import Lattice, Structure
from support import (
    carbon_chain,
    carbon_pair_in_a_box,
    graphene_monolayer,
    graphene_with_c_tilted,
    graphene_with_vacuum_along_a,
    graphene_with_vacuum_along_x,
    graphite,
    molybdenum_disulfide_monolayer,
    rhombohedral_bismuth_selenide,
    rotated_about_x,
    written_cell,
)

from goldilocks_core.advisors.boundary import BoundaryFacts
from goldilocks_core.advisors.electron_count import ElectronCountDecision
from goldilocks_core.advisors.job_resources import JobDecision
from goldilocks_core.advisors.magnetic_config import (
    MagneticConfigFacts,
    OrderingRecord,
)
from goldilocks_core.advisors.nbnd import NbndDecision
from goldilocks_core.advisors.occupations import OccupationsDecision
from goldilocks_core.advisors.parallelisation import ParallelisationDecision
from goldilocks_core.advisors.relax import RelaxOptions, VcRelaxOptions
from goldilocks_core.advisors.vdw_method import VdwFacts
from goldilocks_core.checks import check_all, collect_blocked
from goldilocks_core.resolution import Blocked, Provenance, Resolved, Unavailable

_IRON = Structure(Lattice.cubic(2.87), ["Fe"], [[0.0, 0.0, 0.0]])


def _fixed(tot_magnetization: float | None = None):
    """Fixed occupations pinning ``tot_magnetization`` -- the moment the
    gate checks and the writer renders (#138)."""
    return Resolved(
        OccupationsDecision(
            occupations="fixed",
            smearing_type=None,
            degauss=None,
            tot_magnetization=tot_magnetization,
        ),
        Provenance(source="heuristic"),
    )


_FIXED = _fixed()
_SMEARING = Resolved(
    OccupationsDecision(
        occupations="smearing",
        smearing_type="cold",
        degauss=0.01,
        tot_magnetization=2.5,
    ),
    Provenance(source="heuristic"),
)


def _magnetic(
    spin_polarized: bool,
    structure: Structure = _IRON,
    *,
    spin_orbit_enabled: bool = False,
):
    angles = {"Fe": 0.0} if spin_polarized and spin_orbit_enabled else None
    facts = MagneticConfigFacts(
        cell=structure,
        ordering=OrderingRecord(requested="fm", applied="fm", compensated=False),
        spin_polarized=spin_polarized,
        magnetic_elements=("Fe",) if spin_polarized else (),
        starting_magnetization={"Fe": 0.1} if spin_polarized else None,
        tot_magnetization=None,
        spin_orbit_enabled=spin_orbit_enabled,
        angle1=angles,
        angle2=angles,
    )
    return Resolved(facts, Provenance(source="heuristic"))


def test_collect_blocked_returns_only_blocked_root_causes() -> None:
    causes = collect_blocked(
        Resolved(1, Provenance(source="heuristic")),
        Unavailable(reason="could not tell"),
        Blocked(by="upstream failure"),
    )

    assert causes == ("upstream failure",)


def test_check_all_with_nothing_given_is_ok() -> None:
    report = check_all()

    assert report.ok
    assert report.blocking == ()


def test_check_all_surfaces_a_blocked_field_state() -> None:
    report = check_all(Blocked(by="pseudopotential selection failed"))

    assert not report.ok
    assert report.blocking == ("pseudopotential selection failed",)


def test_fixed_occupations_with_non_integer_moment_blocks() -> None:
    report = check_all(occupations=_fixed(2.5), magnetic=_magnetic(True), purpose="scf")

    assert not report.ok
    assert "integer tot_magnetization" in report.blocking[0]
    assert "2.5" in report.blocking[0]


def test_fixed_occupations_with_missing_moment_blocks() -> None:
    report = check_all(occupations=_FIXED, magnetic=_magnetic(True), purpose="scf")

    assert not report.ok
    assert "none was set" in report.blocking[0]
    assert "occupations='smearing'" in report.blocking[0]


def test_fixed_occupations_with_integer_moment_is_ok() -> None:
    report = check_all(occupations=_fixed(2.0), magnetic=_magnetic(True), purpose="scf")

    assert report.ok


def test_the_moment_rule_reads_the_occupations_decision_not_the_request() -> None:
    """#138: the occupations decision holds the moment the input pins, the
    one the writer renders: the 0 it derives for a compensated ordering
    passes, though no moment was requested."""
    derived = check_all(
        occupations=_fixed(0.0),
        magnetic=_magnetic(True),
        electron_count=_electron_count(48.0),
        purpose="scf",
    )

    assert derived.ok, derived.blocking


def test_smearing_occupations_never_triggers_the_moment_check() -> None:
    report = check_all(occupations=_SMEARING, magnetic=_magnetic(True), purpose="scf")

    assert report.ok


def test_non_magnetic_system_never_triggers_the_moment_check() -> None:
    report = check_all(occupations=_FIXED, magnetic=_magnetic(False), purpose="scf")

    assert report.ok


def test_the_moment_check_is_scoped_to_the_scf_step_only() -> None:
    report = check_all(
        occupations=_fixed(2.5), magnetic=_magnetic(True), purpose="nscf"
    )

    assert report.ok


def test_the_moment_check_also_applies_to_relax_and_vc_relax() -> None:
    """relax/vc-relax re-run their own scf loop at every ionic step (v2
    epic 10, #10) -- unlike nscf, which reads a prior step's already
    -converged density/spin."""
    for purpose in ("relax", "vc-relax"):
        report = check_all(
            occupations=_fixed(2.5), magnetic=_magnetic(True), purpose=purpose
        )

        assert not report.ok, purpose
        assert "integer tot_magnetization" in report.blocking[0]


def test_blocked_occupations_surfaces_once_via_the_generic_scan() -> None:
    report = check_all(occupations=Blocked(by="functional undecided"), purpose="scf")

    assert report.blocking == ("functional undecided",)


def _electron_count(nelec: float):
    return Resolved(
        ElectronCountDecision(nelec=nelec, tot_charge=0.0),
        Provenance(source="heuristic"),
    )


def test_fixed_occupations_non_spin_polarized_odd_electron_count_blocks() -> None:
    """#70: a single boron atom (nelec=3) forced non-metal -- real repro
    that produced a silently-wrong-electron-count scf.in before this
    check existed."""
    report = check_all(
        occupations=_FIXED,
        magnetic=_magnetic(False),
        electron_count=_electron_count(3.0),
        purpose="scf",
    )

    assert not report.ok
    assert "even electron count" in report.blocking[0]
    assert "3.0" in report.blocking[0]


def test_fixed_occupations_non_spin_polarized_even_electron_count_is_ok() -> None:
    report = check_all(
        occupations=_FIXED,
        magnetic=_magnetic(False),
        electron_count=_electron_count(4.0),
        purpose="scf",
    )

    assert report.ok


def test_smearing_occupations_never_triggers_the_electron_count_check() -> None:
    report = check_all(
        occupations=_SMEARING,
        magnetic=_magnetic(False),
        electron_count=_electron_count(3.0),
        purpose="scf",
    )

    assert report.ok


def test_spin_polarized_system_never_triggers_the_electron_count_check() -> None:
    """An odd-electron, spin-polarized system is the sibling
    integer-moment check's territory instead."""
    report = check_all(
        occupations=_fixed(1.0),
        magnetic=_magnetic(True),
        electron_count=_electron_count(3.0),
        purpose="scf",
    )

    assert report.ok


def test_the_electron_count_check_is_scoped_to_the_scf_step_only() -> None:
    report = check_all(
        occupations=_FIXED,
        magnetic=_magnetic(False),
        electron_count=_electron_count(3.0),
        purpose="nscf",
    )

    assert report.ok


def test_the_electron_count_check_also_applies_to_relax_and_vc_relax() -> None:
    for purpose in ("relax", "vc-relax"):
        report = check_all(
            occupations=_FIXED,
            magnetic=_magnetic(False),
            electron_count=_electron_count(3.0),
            purpose=purpose,
        )

        assert not report.ok, purpose
        assert "even electron count" in report.blocking[0]


def test_moment_and_electron_count_must_share_parity() -> None:
    """#70's second finding: an integer tot_magnetization alone is not
    enough -- nelup=(nelec+tot)/2 and neldw=(nelec-tot)/2 both need to
    be integers, which needs matching parity (nelec=3, tot=2 gives
    nelup=2.5/neldw=0.5)."""
    report = check_all(
        occupations=_fixed(2.0),
        magnetic=_magnetic(True),
        electron_count=_electron_count(3.0),
        purpose="scf",
    )

    assert not report.ok
    assert "share parity" in report.blocking[0]


def test_moment_and_electron_count_with_matching_parity_is_ok() -> None:
    report = check_all(
        occupations=_fixed(1.0),
        magnetic=_magnetic(True),
        electron_count=_electron_count(3.0),
        purpose="scf",
    )

    assert report.ok


def test_a_magnetic_spin_orbit_run_needs_no_moment_or_even_count() -> None:
    """#146: spin-orbit coupling runs noncollinear, where pw.x sets
    nspin=4 and lsda=.false. (QE 7.3 PW/src/input.f90:714-736). Its
    "fixed occupations and lsda need tot_magnetization" stop applies only
    with lsda (input.f90:764-781), and its fixed occupations fill
    NINT(nelec) single-electron spinor bands (degspin=1,
    PW/src/iweights.f90:105-107). With moments, time reversal is broken,
    so an odd count can fill whole bands and neither rule applies."""
    report = check_all(
        occupations=_FIXED,
        magnetic=_magnetic(True, spin_orbit_enabled=True),
        electron_count=_electron_count(3.0),
        purpose="scf",
    )

    assert report.ok, report.blocking


def test_a_non_magnetic_spin_orbit_run_needs_an_even_count() -> None:
    """A spin-orbit run without moments keeps time reversal: every band has
    a Kramers partner degenerate with it at the time-reversal-invariant
    k-points (at every k with inversion), so an odd count half-fills a
    degenerate pair, metallic as without spin-orbit coupling. degspin=1
    removes the truncation, not the half-filled band."""
    odd = check_all(
        occupations=_FIXED,
        magnetic=_magnetic(False, spin_orbit_enabled=True),
        electron_count=_electron_count(9.0),
        purpose="scf",
    )
    even = check_all(
        occupations=_FIXED,
        magnetic=_magnetic(False, spin_orbit_enabled=True),
        electron_count=_electron_count(8.0),
        purpose="scf",
    )

    assert odd.blocking == (
        "occupations='fixed' with a non-spin-polarized system requires an "
        "even electron count; got nelec=9.0, which likely means an unpaired "
        "electron -- set spin_polarized=true, or "
        "occupations=smearing/tetrahedra_opt if this is genuinely metallic.",
    )
    assert even.ok, even.blocking


def test_moment_parity_check_is_unaffected_when_electron_count_is_missing() -> None:
    """Existing callers that don't pass electron_count (pre-#70) keep
    exactly their previous behaviour: integer tot_magnetization alone
    is accepted."""
    report = check_all(occupations=_fixed(2.0), magnetic=_magnetic(True), purpose="scf")

    assert report.ok


def _relax(ion_dynamics: str = "bfgs"):
    return Resolved(
        RelaxOptions(ion_dynamics=ion_dynamics), Provenance(source="heuristic")
    )


def _vc_relax(ion_dynamics: str = "bfgs"):
    return Resolved(
        VcRelaxOptions(ion_dynamics=ion_dynamics), Provenance(source="heuristic")
    )


def test_vc_relax_with_bfgs_ion_dynamics_is_ok() -> None:
    report = check_all(relax=_vc_relax("bfgs"), purpose="vc-relax")

    assert report.ok


def test_vc_relax_with_non_bfgs_ion_dynamics_blocks() -> None:
    report = check_all(relax=_vc_relax("damp"), purpose="vc-relax")

    assert not report.ok
    assert "ion_dynamics='bfgs'" in report.blocking[0]
    assert "'damp'" in report.blocking[0]


def test_plain_relax_with_non_bfgs_ion_dynamics_is_unaffected() -> None:
    """The bfgs-only restriction is vc-relax-specific -- QE genuinely
    supports damp/fire for a plain relax."""
    report = check_all(relax=_relax("damp"), purpose="relax")

    assert report.ok


def test_relax_blocked_field_state_surfaces_via_the_generic_scan() -> None:
    report = check_all(relax=Blocked(by="convergence undecided"), purpose="vc-relax")

    assert report.blocking == ("convergence undecided",)


def _boundary(isolation: str, source: str = "heuristic"):
    return Resolved(BoundaryFacts(isolation), Provenance(source=source))


def test_vc_relax_of_an_isolated_system_blocks() -> None:
    """#126: pw.x clears tstress for an isolated system (QE 7.3
    PW/src/input.f90:1361-1365), so BFGS sees a zero cell gradient and
    reports the input cell as converged."""
    report = check_all(boundary=_boundary("isolated"), purpose="vc-relax")

    assert not report.ok
    assert len(report.blocking) == 1
    assert "isolation='isolated'" in report.blocking[0]
    assert "task='relax'" in report.blocking[0]
    assert "isolation='periodic'" in report.blocking[0]


def test_a_human_isolated_override_on_vc_relax_blocks_too() -> None:
    report = check_all(
        boundary=_boundary("isolated", source="human"), purpose="vc-relax"
    )

    assert not report.ok
    assert "task='relax'" in report.blocking[0]


def test_an_isolated_system_without_a_cell_relaxation_passes() -> None:
    for purpose in ("scf", "nscf", "relax"):
        report = check_all(boundary=_boundary("isolated"), purpose=purpose)

        assert report.ok, purpose


@pytest.mark.parametrize("isolation", ["periodic", "slab"])
def test_vc_relax_with_a_boundary_that_computes_stress_passes(isolation: str) -> None:
    report = check_all(boundary=_boundary(isolation), purpose="vc-relax")

    assert report.ok


def test_blocked_boundary_surfaces_once_via_the_generic_scan() -> None:
    report = check_all(boundary=Blocked(by="geometry undecided"), purpose="vc-relax")

    assert report.blocking == ("geometry undecided",)


def _vc_relax_with_cell_dofree(cell_dofree: str):
    return Resolved(VcRelaxOptions(cell_dofree=cell_dofree), Provenance(source="human"))


def _isolation_reasons(isolation: str, structure: Structure) -> tuple[str, ...]:
    return check_all(
        boundary=_boundary(isolation, source="human"),
        magnetic=written_cell(structure),
    ).blocking


@pytest.mark.parametrize(
    "build",
    [graphene_monolayer, molybdenum_disulfide_monolayer],
    ids=lambda build: build.__name__,
)
def test_a_slab_in_the_xy_frame_with_room_for_the_2d_truncation_passes(
    build,
) -> None:
    """#161: QE's assume_isolated='2D' truncates the Coulomb interaction
    beyond half the cell height along z (QE 7.3 PW/src/Coul_cut_2D.f90:77).
    A sheet in a 20 A cell meets every precondition."""
    assert _isolation_reasons("slab", build()) == ()


@pytest.mark.parametrize(
    ("build", "found"),
    [
        (graphite, "no vacuum"),
        (graphene_with_vacuum_along_x, "vacuum along a"),
        (graphene_with_vacuum_along_a, "a or b out of the xy plane and vacuum along a"),
        (
            lambda: rotated_about_x(graphene_monolayer(), 30.0),
            "a or b out of the xy plane",
        ),
        (carbon_chain, "vacuum along a and b"),
        (carbon_pair_in_a_box, "vacuum along a, b and c"),
        (graphene_with_c_tilted, "c off the z axis"),
        (
            lambda: graphene_monolayer(c=11.0),
            "a height of 11.00 Å along z (at least 11.11 Å needed)",
        ),
        (
            lambda: molybdenum_disulfide_monolayer(c=12.0),
            "8.83 Å of vacuum around a slab 3.17 Å thick (at least 13.75 Å needed)",
        ),
    ],
    ids=[
        "bulk",
        "vacuum-along-a",
        "vacuum-vector-out-of-plane",
        "tilted-slab",
        "wire",
        "molecule",
        "c-off-z",
        "below-21-bohr",
        "thick-slab-in-a-short-cell",
    ],
)
def test_a_slab_isolation_without_qe_2d_preconditions_blocks(build, found: str) -> None:
    """'2D' needs the vacuum along c only and a, b in the xy plane (QE 7.3
    PW/src/Coul_cut_2D.f90:72-74), c along z, the only frame in which its
    cutoff factor is exact (Coul_cut_2D.f90:77-81), half the cell height
    above the pseudopotentials' 10 bohr radial cutoff
    (PW/src/init_vloc.f90:62-65), and vacuum 20 bohr wider than the slab
    (PW/Doc/INPUT_PW.def:1668-1675). A 11.00 A cell is below 21 bohr
    (11.11 A); the 3.17 A MoS2 sheet needs 13.75 A of vacuum."""
    reasons = _isolation_reasons("slab", build())

    assert len(reasons) == 1
    assert "isolation='slab'" in reasons[0]
    assert f"the cell to be written has {found}." in reasons[0]
    assert reasons[0].endswith("Set isolation='periodic'.")


def test_the_2d_height_floor_is_21_bohr() -> None:
    """Graphene has no thickness, so 11.12 A of vacuum clears the 20 bohr
    (10.58 A) margin and isolates the height floor: 21 bohr is 11.113 A."""
    assert _isolation_reasons("slab", graphene_monolayer(c=11.12)) == ()
    assert len(_isolation_reasons("slab", graphene_monolayer(c=11.1))) == 1


def test_the_2d_vacuum_margin_is_20_bohr_beyond_the_slab() -> None:
    """The 3.17 A MoS2 sheet needs 3.17 + 10.58 = 13.75 A of vacuum: a
    16.93 A cell leaves 13.76 A, a 16.91 A cell 13.74 A."""
    assert _isolation_reasons("slab", molybdenum_disulfide_monolayer(c=16.93)) == ()
    reasons = _isolation_reasons("slab", molybdenum_disulfide_monolayer(c=16.91))
    assert "13.74 Å of vacuum around a slab 3.17 Å thick" in reasons[0]


def test_an_isolated_system_in_a_box_passes() -> None:
    assert _isolation_reasons("isolated", carbon_pair_in_a_box()) == ()


@pytest.mark.parametrize(
    ("build", "found"),
    [
        (graphite, "no vacuum"),
        (graphene_monolayer, "vacuum only along c"),
        (carbon_chain, "vacuum only along a and b"),
    ],
    ids=["bulk", "slab", "wire"],
)
def test_an_isolated_boundary_on_a_periodic_cell_blocks(build, found: str) -> None:
    """#161: Martyna-Tuckerman is a correction for a molecule or cluster in
    a 3D supercell (QE 7.3 PW/Doc/INPUT_PW.def:1593-1594), and pw.x has no
    1D truncation, so a wire must stay periodic."""
    reasons = _isolation_reasons("isolated", build())

    assert len(reasons) == 1
    assert "isolation='isolated'" in reasons[0]
    assert f"the cell to be written has {found}." in reasons[0]
    assert reasons[0].endswith("Set isolation='periodic'.")


@pytest.mark.parametrize(
    "build", [graphite, carbon_chain, carbon_pair_in_a_box], ids=lambda b: b.__name__
)
def test_a_periodic_boundary_suits_every_cell(build) -> None:
    assert _isolation_reasons("periodic", build()) == ()


@pytest.mark.parametrize("isolation", ["slab", "isolated"])
def test_the_isolation_gate_is_skipped_without_the_written_cell(
    isolation: str,
) -> None:
    assert check_all(boundary=_boundary(isolation)).ok


def test_the_isolation_gate_reads_the_written_cell_whatever_the_source() -> None:
    """The heuristic decides from the input structure; an AFM cell can move
    the vacuum, and the gate judges the cell pw.x reads."""
    report = check_all(
        boundary=_boundary("isolated", source="heuristic"),
        magnetic=written_cell(graphite()),
    )

    assert len(report.blocking) == 1
    assert "isolation='isolated'" in report.blocking[0]


@pytest.mark.parametrize("cell_dofree", ["all", "shape", "z", "fixc", "epitaxial_ab"])
def test_vc_relax_under_the_2d_truncation_needs_a_2d_cell_constraint(
    cell_dofree: str,
) -> None:
    """Under '2D' only in-plane stresses make sense, and QE asks for
    cell_dofree='2Dxy' (QE 7.3 PW/Doc/INPUT_PW.def:1677-1678); pw.x itself
    does not check it."""
    report = check_all(
        boundary=_boundary("slab", source="human"),
        relax=_vc_relax_with_cell_dofree(cell_dofree),
        magnetic=written_cell(graphene_monolayer()),
        purpose="vc-relax",
    )

    assert len(report.blocking) == 1
    assert f"cell_dofree={cell_dofree!r}" in report.blocking[0]
    assert "cell_dofree='2Dxy'" in report.blocking[0]
    assert "isolation='periodic'" in report.blocking[0]


@pytest.mark.parametrize(
    "cell_dofree", ["2Dxy", "2Dshape", "ibrav+2Dxy", "ibrav+2Dshape"]
)
def test_vc_relax_under_the_2d_truncation_with_a_2d_cell_constraint_passes(
    cell_dofree: str,
) -> None:
    report = check_all(
        boundary=_boundary("slab", source="human"),
        relax=_vc_relax_with_cell_dofree(cell_dofree),
        magnetic=written_cell(graphene_monolayer()),
        purpose="vc-relax",
    )

    assert report.ok


def test_a_plain_relax_under_the_2d_truncation_needs_no_cell_constraint() -> None:
    report = check_all(
        boundary=_boundary("slab", source="human"),
        relax=Resolved(RelaxOptions(), Provenance(source="heuristic")),
        magnetic=written_cell(graphene_monolayer()),
        purpose="relax",
    )

    assert report.ok


def test_a_periodic_slab_may_relax_its_whole_cell() -> None:
    report = check_all(
        boundary=_boundary("periodic"),
        relax=_vc_relax_with_cell_dofree("all"),
        magnetic=written_cell(graphene_monolayer()),
        purpose="vc-relax",
    )

    assert report.ok


@pytest.mark.parametrize(
    "build",
    [graphene_with_vacuum_along_x, rhombohedral_bismuth_selenide],
    ids=lambda build: build.__name__,
)
def test_a_2d_cell_constraint_outside_the_xy_frame_blocks(build) -> None:
    """#132: '2Dxy' moves only the x and y components of a and b (QE 7.3
    Modules/cell_base.f90:803-808). With the vacuum along a it frees the
    vacuum vector; with a and b out of the xy plane (rhombohedral Bi2Se3) it
    distorts the cell."""
    report = check_all(
        relax=_vc_relax_with_cell_dofree("2Dxy"),
        magnetic=written_cell(build()),
        purpose="vc-relax",
    )

    assert not report.ok
    assert len(report.blocking) == 1
    assert "cell_dofree='2Dxy'" in report.blocking[0]
    assert "cell_dofree='all'" in report.blocking[0]


@pytest.mark.parametrize(
    ("build", "found"),
    [
        (graphene_with_vacuum_along_x, "vacuum along a"),
        (rhombohedral_bismuth_selenide, "a or b out of the xy plane"),
        (carbon_chain, "vacuum along a and b"),
        (graphene_with_vacuum_along_a, "a or b out of the xy plane and vacuum along a"),
    ],
    ids=lambda value: value.__name__ if callable(value) else None,
)
def test_the_frame_reason_names_what_is_out_of_place(build, found: str) -> None:
    report = check_all(
        relax=_vc_relax_with_cell_dofree("2Dxy"),
        magnetic=written_cell(build()),
        purpose="vc-relax",
    )

    assert f"the cell to be written has {found}." in report.blocking[0]


@pytest.mark.parametrize(
    "cell_dofree", ["2Dxy", "2Dshape", "ibrav+2Dxy", "ibrav+2Dshape"]
)
def test_every_2d_cell_constraint_is_checked_against_the_frame(
    cell_dofree: str,
) -> None:
    report = check_all(
        relax=_vc_relax_with_cell_dofree(cell_dofree),
        magnetic=written_cell(rhombohedral_bismuth_selenide()),
        purpose="vc-relax",
    )

    assert len(report.blocking) == 1
    assert f"cell_dofree={cell_dofree!r}" in report.blocking[0]


@pytest.mark.parametrize(
    "build", [graphite, graphene_monolayer], ids=lambda build: build.__name__
)
def test_a_2d_cell_constraint_in_the_xy_frame_passes(build) -> None:
    """Bulk graphite keeps an explicit '2Dxy' (in-plane relaxation at fixed
    c); the monolayer has its vacuum along c."""
    report = check_all(
        relax=_vc_relax_with_cell_dofree("2Dxy"),
        magnetic=written_cell(build()),
        purpose="vc-relax",
    )

    assert report.ok


@pytest.mark.parametrize("cell_dofree", ["all", "shape", "z"])
def test_other_cell_constraints_are_not_checked_against_the_frame(
    cell_dofree: str,
) -> None:
    report = check_all(
        relax=_vc_relax_with_cell_dofree(cell_dofree),
        magnetic=written_cell(rhombohedral_bismuth_selenide()),
        purpose="vc-relax",
    )

    assert report.ok


def test_the_frame_check_is_skipped_without_the_written_cell() -> None:
    report = check_all(relax=_vc_relax_with_cell_dofree("2Dxy"), purpose="vc-relax")

    assert report.ok


def _vdw(use_vdw: bool, source: str = "heuristic"):
    facts = VdwFacts(use_vdw=use_vdw, method="d3bj" if use_vdw else None)
    return Resolved(facts, Provenance(source=source))


def _functional(label: str):
    return Resolved(label, Provenance(source="human"))


def test_d3_dispersion_with_lda_blocks() -> None:
    """#141: pw.x hands DFT-D3 the pseudopotentials' short functional name
    (QE 7.3 PW/src/input.f90:2039-2041), and the D3(BJ) table has no LDA
    entry, so pw.x would stop with 'functional name unknown'."""
    report = check_all(vdw=_vdw(True, source="human"), functional=_functional("LDA"))

    assert not report.ok
    assert len(report.blocking) == 1
    assert "'LDA'" in report.blocking[0]
    assert "use_vdw=false" in report.blocking[0]


def test_d3_dispersion_with_an_unrecognised_functional_blocks() -> None:
    report = check_all(vdw=_vdw(True, source="llm"), functional=_functional("revPBE"))

    assert not report.ok
    assert "'revPBE'" in report.blocking[0]


def test_d3_dispersion_with_pbe_or_pbesol_passes() -> None:
    for label in ("PBE", "PBEsol"):
        report = check_all(vdw=_vdw(True), functional=_functional(label))

        assert report.ok, label


def test_no_dispersion_with_lda_passes() -> None:
    report = check_all(vdw=_vdw(False), functional=_functional("LDA"))

    assert report.ok


def test_dispersion_check_is_unaffected_when_either_side_is_missing() -> None:
    assert check_all(vdw=_vdw(True)).ok
    assert check_all(functional=_functional("LDA")).ok


def test_blocked_vdw_surfaces_once_via_the_generic_scan() -> None:
    report = check_all(
        vdw=Blocked(by="geometry undecided"), functional=_functional("LDA")
    )

    assert report.blocking == ("geometry undecided",)


def _relax_with_fix_bottom_layers(count: int | None):
    return Resolved(RelaxOptions(fix_bottom_layers=count), Provenance(source="human"))


def _fix_bottom_layers_reasons(count: int | None, structure: Structure):
    return check_all(
        relax=_relax_with_fix_bottom_layers(count),
        magnetic=written_cell(structure),
    ).blocking


@pytest.mark.parametrize(
    "build",
    [graphene_monolayer, molybdenum_disulfide_monolayer, graphene_with_vacuum_along_x],
    ids=lambda build: build.__name__,
)
def test_fix_bottom_layers_on_a_slab_passes(build) -> None:
    """A slab is any cell with vacuum along exactly one lattice vector, in
    whatever orientation: if_pos flags do not depend on the Cartesian
    frame."""
    assert _fix_bottom_layers_reasons(1, build()) == ()


@pytest.mark.parametrize(
    ("build", "found"),
    [
        (graphite, "no vacuum"),
        (carbon_chain, "vacuum along a and b"),
        (carbon_pair_in_a_box, "vacuum along a, b and c"),
    ],
    ids=["bulk-layered-crystal", "wire", "molecule"],
)
def test_fix_bottom_layers_without_exactly_one_vacuum_axis_blocks(
    build, found: str
) -> None:
    """#160: bulk graphite ranks '2d', yet it has no vacuum and so no bottom
    to count layers up from; with fix_bottom_layers=1 the rank-keyed gate
    passed and relax.in froze its z = 1/4 sheet. The gate reads the cell the
    writer renders, not the rank."""
    reasons = _fix_bottom_layers_reasons(1, build())

    assert len(reasons) == 1
    assert reasons[0].startswith("relax.fix_bottom_layers requires a slab")
    assert f"the cell to be written has {found}." in reasons[0]
    assert reasons[0].endswith("Unset fix_bottom_layers.")


def test_fix_bottom_layers_on_a_cell_holding_two_slabs_blocks() -> None:
    """An antiferromagnetic supercell can double a monolayer along its
    normal, putting two sheets and two vacuum gaps along c in the written
    cell. Counted up from one gap, the layers would run through one sheet
    and on into the other, so the gate refuses the cell."""
    reasons = _fix_bottom_layers_reasons(
        1, molybdenum_disulfide_monolayer() * (1, 1, 2)
    )

    assert reasons == (
        "relax.fix_bottom_layers requires a slab: it freezes the bottom atomic "
        "layers counted up from the vacuum, so the cell needs one vacuum gap, "
        "along exactly one lattice vector, but the cell to be written has more "
        "than one vacuum gap along c, so more than one slab. Unset "
        "fix_bottom_layers.",
    )


def test_fix_bottom_layers_may_fix_every_layer_of_the_slab_but_no_more() -> None:
    """D16: the MoS2 sheet has three atomic layers on the written cell (S, Mo,
    S), counted the way the writer counts them, so check() refuses a fourth
    instead of passing a count the writer cannot apply."""
    sheet = molybdenum_disulfide_monolayer()

    assert _fix_bottom_layers_reasons(3, sheet) == ()
    assert _fix_bottom_layers_reasons(4, sheet) == (
        "relax.fix_bottom_layers=4 exceeds the 3 atomic layers of the slab to be "
        "written. Set fix_bottom_layers to at most 3, or unset it.",
    )


def test_fix_bottom_layers_unset_is_never_checked() -> None:
    assert _fix_bottom_layers_reasons(None, graphite()) == ()


def test_the_fix_bottom_layers_gate_is_skipped_without_the_written_cell() -> None:
    assert check_all(relax=_relax_with_fix_bottom_layers(2)).ok


def test_a_blocked_written_cell_surfaces_once_via_the_generic_scan() -> None:
    report = check_all(
        relax=_relax_with_fix_bottom_layers(2),
        magnetic=Blocked(by="magnetic ordering undecided"),
    )

    assert report.blocking == ("magnetic ordering undecided",)


def _job(ntasks: int) -> Resolved[JobDecision]:
    decision = JobDecision(
        partition="scarf",
        nodes=1,
        ntasks=ntasks,
        ntasks_per_node=ntasks,
        walltime_h=168.0,
        max_seconds_by_step={"scf": 574560},
        memory_per_process_mb=12.5,
        memory_total_mb=900.0,
        account=None,
    )
    return Resolved(decision, Provenance(source="heuristic"))


def _parallel(
    npool: int, ndiag: int | None = None
) -> Resolved[ParallelisationDecision]:
    decision = ParallelisationDecision(npool=npool, ndiag=ndiag, kpoint_spin_pairs=64)
    return Resolved(decision, Provenance(source="human"))


def test_npool_dividing_ntasks_is_ok() -> None:
    report = check_all(job=_job(64), parallel=_parallel(8))

    assert report.ok


def test_npool_not_dividing_ntasks_blocks() -> None:
    """Regression for #52: this used to be only a warning, so a bad
    npool reached the SLURM script and failed only once compute was
    already allocated and running."""
    report = check_all(job=_job(64), parallel=_parallel(7))

    assert not report.ok
    assert "npool=7" in report.blocking[0]
    assert "ntasks=64" in report.blocking[0]


def test_npool_check_is_unaffected_when_either_side_is_missing() -> None:
    assert check_all(job=_job(64)).ok
    assert check_all(parallel=_parallel(7)).ok
    assert check_all().ok


def test_npool_blocked_field_state_surfaces_via_the_generic_scan() -> None:
    report = check_all(parallel=Blocked(by="job resources undecided"))

    assert report.blocking == ("job resources undecided",)


def _bands(nbnd: int) -> Resolved[NbndDecision]:
    return Resolved(NbndDecision(nbnd=nbnd), Provenance(source="heuristic"))


@pytest.mark.parametrize(
    ("nbnd", "npool", "ndiag", "side", "largest"),
    [
        (1, 1, 4, 2, 1),  # s > nbnd: set_para_diag's startup abort
        (2, 1, 9, 3, 4),  # LiH, k_grid=[3,3,3], ndiag=9
        (2, 4, 16, 4, 4),  # LiH at the old default -npool 4 -ndiag 16
        (8, 1, 16, 4, 9),  # the C2-box layout, nbnd 8 on a 4x4 grid
        (9, 1, 16, 4, 9),  # nbnd == (s-1)^2: the last empty-column size
        (8, 1, 64, 8, 9),  # the H2O/N2 boxes: -npool 1 -ndiag 64
        (8, 4, 64, 4, 9),  # the grid is capped by 16 processes per pool
        (49, 1, 64, 8, 49),
    ],
)
def test_user_ndiag_whose_grid_cannot_distribute_the_bands_blocks(
    nbnd: int, npool: int, ndiag: int, side: int, largest: int
) -> None:
    """Regression for #133: pw.x stops at startup when the grid side s
    exceeds nbnd (set_para_diag), and leaves a process column without
    rows of a subspace matrix of size n <= (s-1)^2, which ELPA rejects
    and QE does not catch. Both used to pass check() and fail on SCARF."""
    report = check_all(
        job=_job(64), parallel=_parallel(npool, ndiag), nbnd=_bands(nbnd)
    )

    (reason,) = report.blocking
    assert f"ndiag={ndiag}" in reason
    assert f"{side}x{side}" in reason
    assert f"nbnd={nbnd}" in reason
    assert f"ndiag <= {largest}" in reason


def test_ndiag_reason_states_the_grid_the_bound_and_the_remedy() -> None:
    """The C2-box layout from #133: -npool 4 -ndiag 16 on 64 tasks."""
    report = check_all(job=_job(64), parallel=_parallel(4, 16), nbnd=_bands(8))

    assert report.blocking == (
        "ndiag=16 gives a 4x4 diagonalization grid, which cannot distribute "
        "nbnd=8 bands (it needs nbnd > 9); pw.x would abort at startup or "
        "fail in the parallel eigensolver. Unset ndiag so pw.x chooses it "
        "from nbnd, or use ndiag <= 9.",
    )


@pytest.mark.parametrize(
    ("nbnd", "npool", "ndiag"),
    [
        (2, 1, 4),  # s=2 needs nbnd > 1
        (10, 1, 16),  # s=4 needs nbnd > 9
        (50, 1, 64),  # s=8 needs nbnd > 49
        (2, 16, 64),  # 4 processes per pool: s=2
        (8, 1, 10),  # non-square ndiag runs a 3x3 grid
        (1, 1, 1),
    ],
)
def test_user_ndiag_whose_grid_distributes_the_bands_is_ok(
    nbnd: int, npool: int, ndiag: int
) -> None:
    report = check_all(
        job=_job(64), parallel=_parallel(npool, ndiag), nbnd=_bands(nbnd)
    )

    assert report.ok, report.blocking


def test_unset_ndiag_never_blocks() -> None:
    report = check_all(job=_job(64), parallel=_parallel(1), nbnd=_bands(1))

    assert report.ok


def test_ndiag_check_needs_job_parallel_and_bands() -> None:
    assert check_all(parallel=_parallel(1, 64), nbnd=_bands(8)).ok
    assert check_all(job=_job(64), parallel=_parallel(1, 64)).ok
    assert check_all(job=_job(64), nbnd=_bands(8)).ok


def test_unresolved_bands_never_add_the_ndiag_reason() -> None:
    blocked = check_all(
        job=_job(64),
        parallel=_parallel(1, 64),
        nbnd=Blocked(by="electron count undecided"),
    )
    unavailable = check_all(
        job=_job(64),
        parallel=_parallel(1, 64),
        nbnd=Unavailable(reason="no valid band count"),
    )

    assert blocked.blocking == ("electron count undecided",)
    assert unavailable.ok


# #131: pw.x stops in setup when nbnd cannot hold every electron of a spin
# channel (QE 7.3 PW/src/setup.f90:349-356): nbnd >= NINT(nelec/2),
# NINT(nelup) and NINT(neldw), and NINT(nelec) for noncolin. The default
# never falls below these; a human nbnd or extra_bands can.
_TWO_SITE_IRON = Structure(
    Lattice.cubic(2.87), ["Fe", "Fe"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
)


def _band_check(
    nbnd: int,
    nelec: float,
    occupations=_FIXED,
    *,
    spin_polarized: bool = False,
    spin_orbit_enabled: bool = False,
    purpose: str = "scf",
):
    return check_all(
        occupations=occupations,
        magnetic=_magnetic(
            spin_polarized, _TWO_SITE_IRON, spin_orbit_enabled=spin_orbit_enabled
        ),
        electron_count=_electron_count(nelec),
        nbnd=_bands(nbnd),
        purpose=purpose,
    )


def test_too_few_spin_up_bands_for_the_pinned_moment_blocks() -> None:
    """The LiFePO4 deck from #131: tot_magnetization 16 on 192 electrons
    puts 104 in the spin-up channel, and 96 bands made pw.x stop with
    'too few spin up bands' after check() had passed."""
    report = _band_check(96, 192.0, _fixed(16.0), spin_polarized=True)

    assert report.blocking == (
        "nbnd=96 cannot hold 104 spin-up electrons: pw.x stops in setup with "
        "'too few spin up bands'. Set nbnd to at least 104, or unset nbnd and "
        "extra_bands for the default.",
    )


def test_enough_bands_for_the_pinned_moment_pass() -> None:
    assert _band_check(104, 192.0, _fixed(16.0), spin_polarized=True).ok


def test_the_band_reason_names_pw_x_first_error_and_the_count_that_passes() -> None:
    """90 bands fail pw.x's first check, NINT(nelec/2) = 96, but only 104
    pass the spin-up one too."""
    report = _band_check(90, 192.0, _fixed(16.0), spin_polarized=True)

    (reason,) = report.blocking
    assert reason.startswith("nbnd=90 cannot hold 192 electrons, two per band")
    assert "'too few bands'" in reason
    assert "at least 104" in reason


def test_too_few_spin_down_bands_for_a_negative_moment_blocks() -> None:
    report = _band_check(6, 12.0, _fixed(-2.0), spin_polarized=True)

    (reason,) = report.blocking
    assert "'too few spin dw bands'" in reason
    assert "at least 7" in reason


def test_too_few_bands_for_the_electron_count_blocks() -> None:
    """nbnd=2 on 8 electrons passed check() and stopped pw.x with 'too few
    bands'."""
    report = _band_check(2, 8.0)

    (reason,) = report.blocking
    assert reason.startswith("nbnd=2 cannot hold")
    assert "'too few bands'" in reason
    assert "at least 4" in reason


def test_a_noncollinear_run_needs_a_band_per_electron() -> None:
    """Spinor bands hold one electron each (setup.f90:355-356)."""
    blocked = _band_check(13, 14.0, _SMEARING, spin_orbit_enabled=True)
    passed = _band_check(14, 14.0, _SMEARING, spin_orbit_enabled=True)

    (reason,) = blocked.blocking
    assert "'too few bands'" in reason
    assert "at least 14" in reason
    assert passed.ok


@pytest.mark.parametrize("purpose", ["scf", "relax", "vc-relax", "nscf"])
def test_the_band_rule_holds_for_every_purpose(purpose: str) -> None:
    """An nscf step needs every occupied band too, to place the Fermi
    level."""
    report = _band_check(2, 8.0, purpose=purpose)

    assert any("'too few bands'" in reason for reason in report.blocking)


def test_a_blocked_nbnd_is_reported_once() -> None:
    report = check_all(
        occupations=_FIXED,
        magnetic=_magnetic(False, _TWO_SITE_IRON),
        electron_count=_electron_count(8.0),
        nbnd=Blocked(by="electron count undecided"),
    )

    assert report.blocking == ("electron count undecided",)


def test_the_band_rule_needs_occupations_and_the_electron_count() -> None:
    assert check_all(nbnd=_bands(2), electron_count=_electron_count(8.0)).ok
    assert check_all(nbnd=_bands(2), occupations=_FIXED).ok
