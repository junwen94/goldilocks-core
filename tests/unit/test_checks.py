from __future__ import annotations

import pytest
from pymatgen.core import Lattice, Structure
from support import geometry_facts

from goldilocks_core.advisors.boundary import BoundaryFacts
from goldilocks_core.advisors.electron_count import ElectronCountDecision
from goldilocks_core.advisors.job_resources import JobDecision
from goldilocks_core.advisors.magnetic_config import MagneticConfigFacts
from goldilocks_core.advisors.nbnd import NbndDecision
from goldilocks_core.advisors.occupations import OccupationsDecision
from goldilocks_core.advisors.parallelisation import ParallelisationDecision
from goldilocks_core.advisors.relax import RelaxOptions, VcRelaxOptions
from goldilocks_core.checks import check_all, collect_blocked
from goldilocks_core.resolution import Blocked, Provenance, Resolved, Unavailable

_IRON = Structure(Lattice.cubic(2.87), ["Fe"], [[0.0, 0.0, 0.0]])

_FIXED = Resolved(
    OccupationsDecision(occupations="fixed", smearing_type=None, degauss=None),
    Provenance(source="heuristic"),
)
_SMEARING = Resolved(
    OccupationsDecision(occupations="smearing", smearing_type="cold", degauss=0.01),
    Provenance(source="heuristic"),
)


def _magnetic(spin_polarized: bool, tot_magnetization: float | None):
    facts = MagneticConfigFacts(
        relabeled_structure=_IRON,  # not read by any check
        spin_polarized=spin_polarized,
        magnetic_elements=("Fe",) if spin_polarized else (),
        starting_magnetization={"Fe": 0.1} if spin_polarized else None,
        tot_magnetization=tot_magnetization,
        spin_orbit_enabled=False,
        angle1=None,
        angle2=None,
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
    report = check_all(occupations=_FIXED, magnetic=_magnetic(True, 2.5), purpose="scf")

    assert not report.ok
    assert "integer tot_magnetization" in report.blocking[0]
    assert "2.5" in report.blocking[0]


def test_fixed_occupations_with_missing_moment_blocks() -> None:
    report = check_all(
        occupations=_FIXED, magnetic=_magnetic(True, None), purpose="scf"
    )

    assert not report.ok
    assert "none was set" in report.blocking[0]


def test_fixed_occupations_with_integer_moment_is_ok() -> None:
    report = check_all(occupations=_FIXED, magnetic=_magnetic(True, 2.0), purpose="scf")

    assert report.ok


def test_smearing_occupations_never_triggers_the_moment_check() -> None:
    report = check_all(
        occupations=_SMEARING, magnetic=_magnetic(True, 2.5), purpose="scf"
    )

    assert report.ok


def test_non_magnetic_system_never_triggers_the_moment_check() -> None:
    report = check_all(
        occupations=_FIXED, magnetic=_magnetic(False, None), purpose="scf"
    )

    assert report.ok


def test_the_moment_check_is_scoped_to_the_scf_step_only() -> None:
    report = check_all(
        occupations=_FIXED, magnetic=_magnetic(True, 2.5), purpose="nscf"
    )

    assert report.ok


def test_the_moment_check_also_applies_to_relax_and_vc_relax() -> None:
    """relax/vc-relax re-run their own scf loop at every ionic step (v2
    epic 10, #10) -- unlike nscf, which reads a prior step's already
    -converged density/spin."""
    for purpose in ("relax", "vc-relax"):
        report = check_all(
            occupations=_FIXED, magnetic=_magnetic(True, 2.5), purpose=purpose
        )

        assert not report.ok, purpose
        assert "integer tot_magnetization" in report.blocking[0]


def test_blocked_occupations_surfaces_once_via_the_generic_scan() -> None:
    report = check_all(occupations=Blocked(by="functional undecided"), purpose="scf")

    assert report.blocking == ("functional undecided",)


def _electron_count(nelec: float):
    return Resolved(ElectronCountDecision(nelec), Provenance(source="heuristic"))


def test_fixed_occupations_non_spin_polarized_odd_electron_count_blocks() -> None:
    """#70: a single boron atom (nelec=3) forced non-metal -- real repro
    that produced a silently-wrong-electron-count scf.in before this
    check existed."""
    report = check_all(
        occupations=_FIXED,
        magnetic=_magnetic(False, None),
        electron_count=_electron_count(3.0),
        purpose="scf",
    )

    assert not report.ok
    assert "even electron count" in report.blocking[0]
    assert "3.0" in report.blocking[0]


def test_fixed_occupations_non_spin_polarized_even_electron_count_is_ok() -> None:
    report = check_all(
        occupations=_FIXED,
        magnetic=_magnetic(False, None),
        electron_count=_electron_count(4.0),
        purpose="scf",
    )

    assert report.ok


def test_smearing_occupations_never_triggers_the_electron_count_check() -> None:
    report = check_all(
        occupations=_SMEARING,
        magnetic=_magnetic(False, None),
        electron_count=_electron_count(3.0),
        purpose="scf",
    )

    assert report.ok


def test_spin_polarized_system_never_triggers_the_electron_count_check() -> None:
    """An odd-electron, spin-polarized system is the sibling
    integer-moment check's territory instead."""
    report = check_all(
        occupations=_FIXED,
        magnetic=_magnetic(True, 1.0),
        electron_count=_electron_count(3.0),
        purpose="scf",
    )

    assert report.ok


def test_the_electron_count_check_is_scoped_to_the_scf_step_only() -> None:
    report = check_all(
        occupations=_FIXED,
        magnetic=_magnetic(False, None),
        electron_count=_electron_count(3.0),
        purpose="nscf",
    )

    assert report.ok


def test_the_electron_count_check_also_applies_to_relax_and_vc_relax() -> None:
    for purpose in ("relax", "vc-relax"):
        report = check_all(
            occupations=_FIXED,
            magnetic=_magnetic(False, None),
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
        occupations=_FIXED,
        magnetic=_magnetic(True, 2.0),
        electron_count=_electron_count(3.0),
        purpose="scf",
    )

    assert not report.ok
    assert "share parity" in report.blocking[0]


def test_moment_and_electron_count_with_matching_parity_is_ok() -> None:
    report = check_all(
        occupations=_FIXED,
        magnetic=_magnetic(True, 1.0),
        electron_count=_electron_count(3.0),
        purpose="scf",
    )

    assert report.ok


def test_moment_parity_check_is_unaffected_when_electron_count_is_missing() -> None:
    """Existing callers that don't pass electron_count (pre-#70) keep
    exactly their previous behaviour: integer tot_magnetization alone
    is accepted."""
    report = check_all(occupations=_FIXED, magnetic=_magnetic(True, 2.0), purpose="scf")

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


def _boundary(assume_isolated: str, source: str = "heuristic"):
    return Resolved(BoundaryFacts(assume_isolated), Provenance(source=source))


def test_vc_relax_with_martyna_tuckerman_blocks() -> None:
    """#126: pw.x clears tstress for an isolated system (QE 7.3
    PW/src/input.f90:1361-1365), so BFGS sees a zero cell gradient and
    reports the input cell as converged."""
    report = check_all(boundary=_boundary("martyna-tuckerman"), purpose="vc-relax")

    assert not report.ok
    assert len(report.blocking) == 1
    assert "assume_isolated='martyna-tuckerman'" in report.blocking[0]
    assert "task='relax'" in report.blocking[0]
    assert "assume_isolated='none'" in report.blocking[0]


def test_a_human_martyna_tuckerman_override_on_vc_relax_blocks_too() -> None:
    report = check_all(
        boundary=_boundary("martyna-tuckerman", source="human"), purpose="vc-relax"
    )

    assert not report.ok
    assert "task='relax'" in report.blocking[0]


def test_martyna_tuckerman_without_a_cell_relaxation_passes() -> None:
    for purpose in ("scf", "nscf", "relax"):
        report = check_all(boundary=_boundary("martyna-tuckerman"), purpose=purpose)

        assert report.ok, purpose


def test_vc_relax_with_a_periodic_boundary_passes() -> None:
    report = check_all(boundary=_boundary("none"), purpose="vc-relax")

    assert report.ok


def test_blocked_boundary_surfaces_once_via_the_generic_scan() -> None:
    report = check_all(boundary=Blocked(by="geometry undecided"), purpose="vc-relax")

    assert report.blocking == ("geometry undecided",)


def _geometry(dimensionality: str):
    return Resolved(geometry_facts(dimensionality), Provenance(source="heuristic"))


def _relax_with_fix_bottom_layers(count: int | None):
    return Resolved(
        RelaxOptions(fix_bottom_layers=count), Provenance(source="heuristic")
    )


def test_fix_bottom_layers_on_a_2d_structure_is_ok() -> None:
    report = check_all(relax=_relax_with_fix_bottom_layers(2), geometry=_geometry("2d"))

    assert report.ok


def test_fix_bottom_layers_unset_is_unaffected_by_geometry() -> None:
    report = check_all(
        relax=_relax_with_fix_bottom_layers(None), geometry=_geometry("3d")
    )

    assert report.ok


def test_fix_bottom_layers_on_a_3d_structure_blocks() -> None:
    report = check_all(relax=_relax_with_fix_bottom_layers(2), geometry=_geometry("3d"))

    assert not report.ok
    assert "2D slab structure" in report.blocking[0]
    assert "dimensionality='3d'" in report.blocking[0]


def test_fix_bottom_layers_with_no_geometry_at_all_blocks() -> None:
    report = check_all(relax=_relax_with_fix_bottom_layers(2))

    assert not report.ok
    assert "geometry classification is not available" in report.blocking[0]


def test_fix_bottom_layers_with_unresolved_geometry_blocks() -> None:
    report = check_all(
        relax=_relax_with_fix_bottom_layers(2),
        geometry=Unavailable(reason="dimensionality classification failed"),
    )

    assert not report.ok
    assert "geometry classification is not available" in report.blocking[0]


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
