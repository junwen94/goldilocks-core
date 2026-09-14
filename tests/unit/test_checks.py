from __future__ import annotations

from pymatgen.core import Lattice, Structure

from goldilocks_core.advisors.magnetic_config import MagneticConfigFacts
from goldilocks_core.advisors.occupations import OccupationsDecision
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
