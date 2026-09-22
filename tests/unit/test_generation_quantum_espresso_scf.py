from __future__ import annotations

import dataclasses

import pytest
from pymatgen.core import Lattice, Species, Structure

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
from goldilocks_core.advisors.vdw_method import VdwFacts
from goldilocks_core.generation.errors import GenerationError
from goldilocks_core.generation.quantum_espresso.scf import write_qe_scf
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


def _magnetic(
    structure,
    *,
    spin_polarized: bool = False,
    starting_magnetization: dict[str, float] | None = None,
    tot_magnetization: float | None = None,
    spin_orbit_enabled: bool = False,
    angle1: dict[str, float] | None = None,
    angle2: dict[str, float] | None = None,
) -> MagneticConfigFacts:
    return MagneticConfigFacts(
        relabeled_structure=structure,
        spin_polarized=spin_polarized,
        magnetic_elements=(),
        starting_magnetization=starting_magnetization,
        tot_magnetization=tot_magnetization,
        spin_orbit_enabled=spin_orbit_enabled,
        angle1=angle1,
        angle2=angle2,
    )


def _system(
    structure,
    pseudo_metadata_factory,
    *,
    functional: str = "PBEsol",
    magnetic: MagneticConfigFacts | None = None,
    vdw: VdwFacts | None = None,
    hubbard: HubbardUDecision | None = None,
    boundary: BoundaryFacts | None = None,
) -> SystemSettings:
    elements = sorted({site.specie.symbol for site in structure})
    return SystemSettings(
        functional=functional,
        cutoffs=CutoffsDecision(ecutwfc_ry=30.0, ecutrho_ry=120.0),
        # z_valence=4.0 per atom below and nelec=4.0 * len(structure) agree
        # exactly, so _charge_keywords's derived tot_charge is 0.0 (omitted)
        # unless a test overrides electron_count itself -- these tests are
        # about other keywords, not #34's tot_charge translation.
        electron_count=ElectronCountDecision(nelec=4.0 * len(structure)),
        pseudopotentials=tuple(
            pseudo_metadata_factory(element, functional=functional, z_valence=4.0)
            for element in elements
        ),
        magnetic=magnetic or _magnetic(structure),
        vdw=vdw or VdwFacts(use_vdw=False, method=None),
        hubbard=hubbard or HubbardUDecision(plan="not_needed", u_by_element={}),
        boundary=boundary or BoundaryFacts(assume_isolated="none"),
    )


_UNSET = object()


def _step(
    *,
    occupations: OccupationsDecision | None = _UNSET,
    k_sampling: KSamplingDecision | None = _UNSET,
    nbnd: NbndDecision | None = _UNSET,
    convergence: ConvergenceDecision | None = _UNSET,
    parallel: ParallelisationDecision | None = _UNSET,
    relax=None,
    disk_io: str | None = None,
) -> PwSettings:
    return PwSettings(
        disk_io=disk_io,
        occupations=(
            OccupationsDecision(occupations="fixed", smearing_type=None, degauss=None)
            if occupations is _UNSET
            else occupations
        ),
        k_sampling=(
            KSamplingDecision(mesh=(3, 3, 3), shift=(0, 0, 0), k_distance=0.3)
            if k_sampling is _UNSET
            else k_sampling
        ),
        n_irr_k=4,
        nbnd=NbndDecision(nbnd=4) if nbnd is _UNSET else nbnd,
        convergence=(
            ConvergenceDecision(
                conv_thr=1e-8,
                etot_conv_thr=1e-5,
                mixing_beta=0.4,
                electron_maxstep=80,
                mixing_mode="plain",
                mixing_fixed_ns=None,
            )
            if convergence is _UNSET
            else convergence
        ),
        parallel=(
            ParallelisationDecision(npool=2, ndiag=None)
            if parallel is _UNSET
            else parallel
        ),
        relax=relax,
    )


def test_nelec_override_translates_to_qe_s_real_tot_charge_keyword(
    silicon_structure, pseudo_metadata_factory
) -> None:
    """Regression for #34 (v2 epic 9, #9): QE's real &SYSTEM namelist has
    no ``nelec`` input keyword at all (confirmed against ASE's own
    ALL_KEYS['pw']['system'] registry -- only ``tot_charge`` exists) --
    a resolved ``electron_count.nelec`` different from the natural,
    pseudopotential-summed count must translate to ``tot_charge``, the
    deviation from neutral, not be dropped or emitted verbatim under a
    keyword QE would reject."""
    system = _system(silicon_structure, pseudo_metadata_factory)
    system = dataclasses.replace(
        system, electron_count=ElectronCountDecision(nelec=3.0)
    )
    step = _step()

    content = write_qe_scf(system, step, _JOB, _CTX)[0].files["scf.in"]

    assert "tot_charge       = 1.0" in content
    assert "nelec" not in content


def test_no_tot_charge_keyword_when_nelec_matches_the_natural_count(
    silicon_structure, pseudo_metadata_factory
) -> None:
    """The default heuristic path resolves nelec to exactly the natural
    (pseudopotential-summed) count -- tot_charge would be 0.0 and is
    pure noise, so it must not appear at all."""
    system = _system(silicon_structure, pseudo_metadata_factory)
    step = _step()

    content = write_qe_scf(system, step, _JOB, _CTX)[0].files["scf.in"]

    assert "tot_charge" not in content


def test_writes_one_scf_step_with_pure_translated_content(
    silicon_structure, pseudo_metadata_factory
) -> None:
    system = _system(silicon_structure, pseudo_metadata_factory)
    step = _step()

    steps = write_qe_scf(system, step, _JOB, _CTX)

    assert len(steps) == 1
    rendered = steps[0]
    assert rendered.name == "scf"
    assert rendered.executable == "pw.x"
    assert rendered.args == ("-npool", "2", "-in", "scf.in")
    assert rendered.stdout == "scf.out"
    content = rendered.files["scf.in"]
    assert "&CONTROL" in content
    assert "calculation      = 'scf'" in content
    assert "prefix           = 'pwscf'" in content
    assert "outdir           = './out'" in content
    assert "pseudo_dir       = './pseudo'" in content
    assert "ecutwfc          = 30.0" in content
    assert "ecutrho          = 120.0" in content
    assert "max_seconds      = 574560" in content
    assert "ATOMIC_SPECIES" in content
    assert "Si  " in content and "Si.UPF" in content
    assert "CELL_PARAMETERS angstrom" in content
    assert "ATOMIC_POSITIONS crystal" in content
    assert "K_POINTS automatic" in content
    assert "3  3  3  0  0  0" in content
    assert "&IONS" not in content
    assert "&CELL" not in content
    assert "nspin" not in content
    assert "vdw_corr" not in content


def test_purpose_nscf_renames_the_step_and_the_calculation_keyword(
    silicon_structure, pseudo_metadata_factory
) -> None:
    """v2 epic 9 (#9): DOS's nscf step is the same writer, a different
    ``purpose`` -- QE's own documented ``calculation='nscf'`` value, not
    a new scientific decision this writer makes."""
    system = _system(silicon_structure, pseudo_metadata_factory)
    step = _step()

    steps = write_qe_scf(system, step, _JOB, _CTX, purpose="nscf")

    assert len(steps) == 1
    rendered = steps[0]
    assert rendered.name == "nscf"
    assert rendered.args == ("-npool", "2", "-in", "nscf.in")
    assert rendered.stdout == "nscf.out"
    content = rendered.files["nscf.in"]
    assert "calculation      = 'nscf'" in content


def test_ndiag_is_appended_to_args_when_resolved(
    silicon_structure, pseudo_metadata_factory
) -> None:
    system = _system(silicon_structure, pseudo_metadata_factory)
    step = _step(parallel=ParallelisationDecision(npool=4, ndiag=16))

    steps = write_qe_scf(system, step, _JOB, _CTX)

    assert steps[0].args == ("-npool", "4", "-ndiag", "16", "-in", "scf.in")


def test_spin_polarized_starting_magnetization_survives_unclobbered(
    pseudo_metadata_factory,
) -> None:
    """The A2 regression this whole rewrite exists to prevent: a real,
    non-zero starting_magnetization must reach the rendered file exactly,
    never silently reset to 0.0 by anything downstream."""
    structure = Structure(Lattice.cubic(2.87), ["Fe"], [[0.0, 0.0, 0.0]])
    magnetic = _magnetic(
        structure, spin_polarized=True, starting_magnetization={"Fe": 0.6923}
    )
    system = _system(structure, pseudo_metadata_factory, magnetic=magnetic)
    step = _step()

    content = write_qe_scf(system, step, _JOB, _CTX)[0].files["scf.in"]

    assert "nspin            = 2" in content
    assert "starting_magnetization(1) = 0.6923" in content
    assert "0.0" not in content.split("starting_magnetization")[1].split("\n")[0]


def test_tot_magnetization_emitted_only_without_spin_orbit(
    pseudo_metadata_factory,
) -> None:
    structure = Structure(Lattice.cubic(2.87), ["Fe"], [[0.0, 0.0, 0.0]])
    magnetic = _magnetic(
        structure,
        spin_polarized=True,
        starting_magnetization={"Fe": 0.6923},
        tot_magnetization=2.0,
    )
    system = _system(structure, pseudo_metadata_factory, magnetic=magnetic)

    content = write_qe_scf(system, _step(), _JOB, _CTX)[0].files["scf.in"]

    assert "tot_magnetization = 2.0" in content


def test_tot_magnetization_with_spin_orbit_is_rejected(pseudo_metadata_factory) -> None:
    structure = Structure(Lattice.cubic(2.87), ["Fe"], [[0.0, 0.0, 0.0]])
    magnetic = _magnetic(
        structure,
        spin_polarized=True,
        starting_magnetization={"Fe": 0.6923},
        tot_magnetization=2.0,
        spin_orbit_enabled=True,
        angle1={"Fe": 0.0},
        angle2={"Fe": 0.0},
    )
    system = _system(structure, pseudo_metadata_factory, magnetic=magnetic)

    with pytest.raises(GenerationError, match="tot_magnetization"):
        write_qe_scf(system, _step(), _JOB, _CTX)


def test_spin_orbit_magnetic_emits_noncolin_lspinorb_and_angles(
    pseudo_metadata_factory,
) -> None:
    structure = Structure(Lattice.cubic(2.87), ["Fe"], [[0.0, 0.0, 0.0]])
    magnetic = _magnetic(
        structure,
        spin_polarized=True,
        starting_magnetization={"Fe": 0.6923},
        spin_orbit_enabled=True,
        angle1={"Fe": 0.0},
        angle2={"Fe": 0.0},
    )
    system = _system(structure, pseudo_metadata_factory, magnetic=magnetic)

    content = write_qe_scf(system, _step(), _JOB, _CTX)[0].files["scf.in"]

    assert "noncolin         = .true." in content
    assert "lspinorb         = .true." in content
    assert "starting_magnetization(1) = 0.6923" in content
    assert "angle1(1)" in content
    assert "angle2(1)" in content
    assert "nspin" not in content
    # QE 7.3 hard-rejects the Gamma-only *algorithm* (real-valued
    # wavefunctions) together with noncolin (PW/src/setup.f90) -- sampling
    # at Gamma is fine, but it must never be spelled as the special
    # "K_POINTS gamma" card, only "K_POINTS automatic" with an explicit
    # 1 1 1 grid (v2 epic 9, #9; goldilocks-core-design.md's own 2026-09-09
    # correction on this exact point). v2 never emits the "gamma" card at
    # all, under any settings -- this pins that invariant specifically for
    # the one combination QE would otherwise refuse to run.
    assert "K_POINTS automatic" in content
    assert "K_POINTS gamma" not in content


def test_afm_relabeled_species_reach_atomic_species_and_positions(
    pseudo_metadata_factory,
) -> None:
    """Regression for the KeyError crash found while wrapping up v2 epic
    9 (#9), filed as #27: an AFM-relabeled structure carries two QE
    species (``Fe1``/``Fe2``) for one real element -- ``elements``
    (real chemistry, for pseudopotential lookup) and the QE species
    list (``site.label``, for ATOMIC_SPECIES/ATOMIC_POSITIONS/
    ``species_index``) must stay two different lists, not one."""
    original = Structure(
        Lattice.cubic(4.3), ["Fe", "O"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    )
    relabeled = Structure(
        Lattice.cubic(4.3),
        [
            Species("Fe", spin=5.0),
            Species("Fe", spin=-5.0),
            "O",
            "O",
        ],
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.5],
            [0.75, 0.5, 0.75],
            [0.25, 0.5, 0.25],
        ],
        labels=["Fe1", "Fe2", "O", "O"],
    )
    magnetic = _magnetic(
        relabeled,
        spin_polarized=True,
        starting_magnetization={"Fe1": 0.3125, "Fe2": -0.3125, "O": 0.1},
    )
    system = _system(original, pseudo_metadata_factory, magnetic=magnetic)

    content = write_qe_scf(system, _step(), _JOB, _CTX)[0].files["scf.in"]

    assert "ntyp             = 3" in content
    assert "nat              = 4" in content
    species = content.split("ATOMIC_SPECIES\n")[1].split("\n\n")[0]
    assert (
        species == "  Fe1  55.845  Fe.UPF\n  Fe2  55.845  Fe.UPF\n  O  15.9994  O.UPF"
    )
    assert "starting_magnetization(1) = 0.3125" in content
    assert "starting_magnetization(2) = -0.3125" in content
    assert "starting_magnetization(3) = 0.1" in content
    positions = content.split("ATOMIC_POSITIONS")[1]
    assert "Fe1  0" in positions
    assert "Fe2  0.5" in positions


def test_vdw_method_translates_to_qe_keyword(
    silicon_structure, pseudo_metadata_factory
) -> None:
    system = _system(
        silicon_structure,
        pseudo_metadata_factory,
        vdw=VdwFacts(use_vdw=True, method="d3bj"),
    )

    content = write_qe_scf(system, _step(), _JOB, _CTX)[0].files["scf.in"]

    assert "vdw_corr         = 'grimme-d3'" in content
    assert "dftd3_version    = 4" in content


def test_boundary_assume_isolated_emitted_when_not_none(
    silicon_structure, pseudo_metadata_factory
) -> None:
    system = _system(
        silicon_structure,
        pseudo_metadata_factory,
        boundary=BoundaryFacts(assume_isolated="martyna-tuckerman"),
    )

    content = write_qe_scf(system, _step(), _JOB, _CTX)[0].files["scf.in"]

    assert "assume_isolated  = 'martyna-tuckerman'" in content


def test_smearing_occupations_emit_smearing_type_and_degauss(
    silicon_structure, pseudo_metadata_factory
) -> None:
    system = _system(silicon_structure, pseudo_metadata_factory)
    step = _step(
        occupations=OccupationsDecision(
            occupations="smearing", smearing_type="cold", degauss=0.01
        )
    )

    content = write_qe_scf(system, step, _JOB, _CTX)[0].files["scf.in"]

    assert "occupations      = 'smearing'" in content
    assert "smearing         = 'cold'" in content
    assert "degauss          = 0.01" in content


def test_hubbard_table_plan_renders_the_qe_hubbard_card(
    silicon_structure, pseudo_metadata_factory
) -> None:
    """QE >= 7.1's onsite U line is species-label-keyed plus its Hubbard
    manifold, not atom-index-keyed -- verified against QE 7.3's own
    Doc/Hubbard_input.tex ("U Mn-3d 5.0") and Modules/read_cards.f90's
    card_hubbard parsing (#88); this confirms the writer renders exactly
    that for plan="table", the common package-default-U case."""
    system = _system(
        silicon_structure,
        pseudo_metadata_factory,
        hubbard=HubbardUDecision(plan="table", u_by_element={"Si": 3.0}),
    )

    content = write_qe_scf(system, _step(), _JOB, _CTX)[0].files["scf.in"]

    assert "HUBBARD (ortho-atomic)" in content
    assert "U  Si-3p  3" in content


def test_hubbard_calibration_needed_plan_still_raises_generation_error(
    silicon_structure, pseudo_metadata_factory
) -> None:
    """plan="self_consistent_calibration_needed" needs an hp.x
    calibration input this codebase has never written (#88) -- real,
    separate, larger scope than the table-plan card above."""
    system = _system(
        silicon_structure,
        pseudo_metadata_factory,
        hubbard=HubbardUDecision(
            plan="self_consistent_calibration_needed", u_by_element={}
        ),
    )

    with pytest.raises(GenerationError, match="calibration"):
        write_qe_scf(system, _step(), _JOB, _CTX)


@pytest.mark.parametrize(
    "field", ["occupations", "k_sampling", "nbnd", "convergence", "parallel"]
)
def test_missing_required_step_setting_raises_generation_error(
    silicon_structure, pseudo_metadata_factory, field
) -> None:
    system = _system(silicon_structure, pseudo_metadata_factory)
    step = _step(**{field: None})

    with pytest.raises(GenerationError, match=field):
        write_qe_scf(system, step, _JOB, _CTX)


def test_relax_settings_on_an_scf_step_is_rejected(
    silicon_structure, pseudo_metadata_factory
) -> None:
    from goldilocks_core.advisors.relax import RelaxOptions

    system = _system(silicon_structure, pseudo_metadata_factory)
    step = _step(relax=RelaxOptions())

    with pytest.raises(GenerationError, match="relax"):
        write_qe_scf(system, step, _JOB, _CTX)


def test_missing_pseudopotential_element_raises_generation_error(
    pseudo_metadata_factory,
) -> None:
    structure = Structure(
        Lattice.cubic(5.64), ["Na", "Cl"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    )
    system = SystemSettings(
        functional="PBEsol",
        cutoffs=CutoffsDecision(ecutwfc_ry=30.0, ecutrho_ry=120.0),
        electron_count=ElectronCountDecision(nelec=8.0),
        pseudopotentials=(pseudo_metadata_factory("Na"),),
        magnetic=_magnetic(structure),
        vdw=VdwFacts(use_vdw=False, method=None),
        hubbard=HubbardUDecision(plan="not_needed", u_by_element={}),
        boundary=BoundaryFacts(assume_isolated="none"),
    )

    with pytest.raises(GenerationError, match="missing"):
        write_qe_scf(system, _step(), _JOB, _CTX)


def test_pseudopotential_functional_mismatch_raises_generation_error(
    silicon_structure, pseudo_metadata_factory
) -> None:
    system = SystemSettings(
        functional="PBEsol",
        cutoffs=CutoffsDecision(ecutwfc_ry=30.0, ecutrho_ry=120.0),
        electron_count=ElectronCountDecision(nelec=4.0),
        pseudopotentials=(pseudo_metadata_factory("Si", functional="LDA"),),
        magnetic=_magnetic(silicon_structure),
        vdw=VdwFacts(use_vdw=False, method=None),
        hubbard=HubbardUDecision(plan="not_needed", u_by_element={}),
        boundary=BoundaryFacts(assume_isolated="none"),
    )

    with pytest.raises(GenerationError, match="functional mismatch"):
        write_qe_scf(system, _step(), _JOB, _CTX)


def test_disordered_structure_is_rejected(pseudo_metadata_factory) -> None:
    structure = Structure(
        Lattice.cubic(4.0),
        [{"Si": 0.5, "Ge": 0.5}],
        [[0.0, 0.0, 0.0]],
    )
    magnetic = _magnetic(structure)
    system = SystemSettings(
        functional="PBEsol",
        cutoffs=CutoffsDecision(ecutwfc_ry=30.0, ecutrho_ry=120.0),
        electron_count=ElectronCountDecision(nelec=4.0),
        pseudopotentials=(pseudo_metadata_factory("Si"), pseudo_metadata_factory("Ge")),
        magnetic=magnetic,
        vdw=VdwFacts(use_vdw=False, method=None),
        hubbard=HubbardUDecision(plan="not_needed", u_by_element={}),
        boundary=BoundaryFacts(assume_isolated="none"),
    )

    with pytest.raises(GenerationError, match="disordered"):
        write_qe_scf(system, _step(), _JOB, _CTX)
