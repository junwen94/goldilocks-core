from __future__ import annotations

import pytest

from goldilocks_core.advisors.boundary import BoundaryFacts
from goldilocks_core.advisors.convergence import ConvergenceDecision
from goldilocks_core.advisors.cutoffs import CutoffsDecision
from goldilocks_core.advisors.electron_count import ElectronCountDecision
from goldilocks_core.advisors.hubbard_u import HubbardUDecision
from goldilocks_core.advisors.job_resources import JobDecision
from goldilocks_core.advisors.k_sampling import KSamplingDecision
from goldilocks_core.advisors.magnetic_config import MagneticConfigFacts
from goldilocks_core.advisors.nbnd import NbndDecision
from goldilocks_core.advisors.occupations import OccupationsDecision
from goldilocks_core.advisors.parallelisation import ParallelisationDecision
from goldilocks_core.advisors.relax import RelaxOptions, VcRelaxOptions
from goldilocks_core.advisors.vdw_method import VdwFacts
from goldilocks_core.generation.errors import GenerationError
from goldilocks_core.generation.quantum_espresso.relax import write_qe_relax
from goldilocks_core.step_settings import PwSettings
from goldilocks_core.steps import SharedContext
from goldilocks_core.system_settings import SystemSettings

_CTX = SharedContext(prefix="pwscf", outdir="./out", pseudo_dir="./pseudo")
_JOB = JobDecision(
    partition="scarf",
    nodes=1,
    ntasks=64,
    ntasks_per_node=64,
    walltime_h=168.0,
    max_seconds=574560,
    account=None,
)
_UNSET = object()


def _system(structure, pseudo_metadata_factory, *, functional: str = "PBEsol"):
    elements = sorted({site.specie.symbol for site in structure})
    return SystemSettings(
        functional=functional,
        cutoffs=CutoffsDecision(ecutwfc_ry=30.0, ecutrho_ry=120.0),
        electron_count=ElectronCountDecision(nelec=4.0 * len(structure)),
        pseudopotentials=tuple(
            pseudo_metadata_factory(element, functional=functional, z_valence=4.0)
            for element in elements
        ),
        magnetic=MagneticConfigFacts(
            relabeled_structure=structure,
            spin_polarized=False,
            magnetic_elements=(),
            starting_magnetization=None,
            tot_magnetization=None,
            spin_orbit_enabled=False,
            angle1=None,
            angle2=None,
        ),
        vdw=VdwFacts(use_vdw=False, method=None),
        hubbard=HubbardUDecision(plan="not_needed", u_by_element={}),
        boundary=BoundaryFacts(assume_isolated="none"),
    )


def _step(*, relax, convergence: ConvergenceDecision | None = _UNSET) -> PwSettings:
    return PwSettings(
        disk_io=None,
        occupations=OccupationsDecision(
            occupations="fixed", smearing_type=None, degauss=None
        ),
        k_sampling=KSamplingDecision(mesh=(3, 3, 3), shift=(0, 0, 0), k_distance=0.3),
        n_irr_k=4,
        nbnd=NbndDecision(nbnd=4),
        convergence=(
            ConvergenceDecision(
                conv_thr=1e-8,
                etot_conv_thr=4e-5,
                mixing_beta=0.4,
                electron_maxstep=80,
                mixing_mode="plain",
                mixing_fixed_ns=None,
            )
            if convergence is _UNSET
            else convergence
        ),
        parallel=ParallelisationDecision(npool=2, ndiag=None),
        relax=relax,
    )


def _line_with(content: str, needle: str) -> str:
    (line,) = (line for line in content.splitlines() if needle in line)
    return line


def test_writes_one_relax_step_with_an_ions_namelist_and_no_cell(
    silicon_structure, pseudo_metadata_factory
) -> None:
    system = _system(silicon_structure, pseudo_metadata_factory)
    step = _step(relax=RelaxOptions())

    steps = write_qe_relax(system, step, _JOB, _CTX, "relax")

    assert len(steps) == 1
    rendered = steps[0]
    assert rendered.name == "relax"
    assert rendered.args == ("-npool", "2", "-in", "relax.in")
    assert rendered.stdout == "relax.out"
    content = rendered.files["relax.in"]
    assert "calculation      = 'relax'" in content
    assert "&IONS" in content
    assert "&CELL" not in content
    assert "ion_dynamics" in content
    assert "forc_conv_thr" in content
    assert "etot_conv_thr" in content
    assert "nstep" in content
    assert "trust_radius_max" in content
    assert "trust_radius_min" in content
    assert "trust_radius_ini" in content
    # shared with scf: same system/electrons/cards
    assert "ecutwfc          = 30.0" in content
    assert "ATOMIC_SPECIES" in content
    assert "CELL_PARAMETERS angstrom" in content
    assert "ATOMIC_POSITIONS crystal" in content
    assert "K_POINTS automatic" in content


def test_writes_vc_relax_step_with_ions_and_cell_namelists(
    silicon_structure, pseudo_metadata_factory
) -> None:
    system = _system(silicon_structure, pseudo_metadata_factory)
    step = _step(relax=VcRelaxOptions())

    steps = write_qe_relax(system, step, _JOB, _CTX, "vc-relax")

    assert len(steps) == 1
    rendered = steps[0]
    assert rendered.name == "vc-relax"
    assert rendered.args == ("-npool", "2", "-in", "vc-relax.in")
    content = rendered.files["vc-relax.in"]
    assert "calculation      = 'vc-relax'" in content
    assert "&IONS" in content
    assert "&CELL" in content
    assert "cell_dynamics" in content
    assert "cell_dofree" in content
    assert "press" in content
    assert "press_conv_thr" in content
    assert "cell_factor" in content


def test_etot_conv_thr_matches_the_resolved_convergence_decision(
    silicon_structure, pseudo_metadata_factory
) -> None:
    system = _system(silicon_structure, pseudo_metadata_factory)
    relax = RelaxOptions(etot_conv_thr=4e-5)
    step = _step(relax=relax)

    content = write_qe_relax(system, step, _JOB, _CTX, "relax")[0].files["relax.in"]

    assert "4e-05" in _line_with(content, "etot_conv_thr")


def test_trust_radius_omitted_when_ion_dynamics_is_not_bfgs(
    silicon_structure, pseudo_metadata_factory
) -> None:
    system = _system(silicon_structure, pseudo_metadata_factory)
    step = _step(relax=RelaxOptions(ion_dynamics="damp"))

    content = write_qe_relax(system, step, _JOB, _CTX, "relax")[0].files["relax.in"]

    assert "ion_dynamics     = 'damp'" in content
    assert "trust_radius_max" not in content
    assert "trust_radius_min" not in content
    assert "trust_radius_ini" not in content


def test_remove_rigid_rot_true_is_rendered(
    silicon_structure, pseudo_metadata_factory
) -> None:
    system = _system(silicon_structure, pseudo_metadata_factory)
    step = _step(relax=RelaxOptions(remove_rigid_rot=True))

    content = write_qe_relax(system, step, _JOB, _CTX, "relax")[0].files["relax.in"]

    assert ".true." in _line_with(content, "remove_rigid_rot")


def test_missing_relax_field_raises(silicon_structure, pseudo_metadata_factory) -> None:
    system = _system(silicon_structure, pseudo_metadata_factory)
    step = _step(relax=None)

    with pytest.raises(GenerationError, match="relax"):
        write_qe_relax(system, step, _JOB, _CTX, "relax")


def test_vc_relax_purpose_with_plain_relax_options_raises(
    silicon_structure, pseudo_metadata_factory
) -> None:
    system = _system(silicon_structure, pseudo_metadata_factory)
    step = _step(relax=RelaxOptions())

    with pytest.raises(GenerationError, match="VcRelaxOptions"):
        write_qe_relax(system, step, _JOB, _CTX, "vc-relax")


def test_relax_purpose_with_vc_relax_options_raises(
    silicon_structure, pseudo_metadata_factory
) -> None:
    system = _system(silicon_structure, pseudo_metadata_factory)
    step = _step(relax=VcRelaxOptions())

    with pytest.raises(GenerationError, match="RelaxOptions"):
        write_qe_relax(system, step, _JOB, _CTX, "relax")


def test_missing_convergence_still_raises_same_as_scf(
    silicon_structure, pseudo_metadata_factory
) -> None:
    system = _system(silicon_structure, pseudo_metadata_factory)
    step = _step(relax=RelaxOptions(), convergence=None)

    with pytest.raises(GenerationError, match="convergence"):
        write_qe_relax(system, step, _JOB, _CTX, "relax")
