from __future__ import annotations

import dataclasses

import pytest
from pymatgen.core import Lattice, Species, Structure
from support import carbon_pair_in_a_box, graphene_monolayer

from goldilocks_core.advisors.boundary import BoundaryFacts
from goldilocks_core.advisors.convergence import ConvergenceDecision
from goldilocks_core.advisors.cutoffs import CutoffsDecision
from goldilocks_core.advisors.electron_count import ElectronCountDecision
from goldilocks_core.advisors.hubbard_u import HubbardU, HubbardUDecision
from goldilocks_core.advisors.k_sampling import KSamplingDecision
from goldilocks_core.advisors.magnetic_config import (
    MagneticConfigFacts,
    OrderingRecord,
)
from goldilocks_core.advisors.nbnd import NbndDecision
from goldilocks_core.advisors.occupations import OccupationsDecision
from goldilocks_core.advisors.parallelisation import (
    ParallelisationDecision,
    qe_spin_layout,
)
from goldilocks_core.advisors.vdw_method import VdwFacts
from goldilocks_core.generation.errors import GenerationError
from goldilocks_core.generation.quantum_espresso.scf import write_qe_scf
from goldilocks_core.step_settings import PwSettings
from goldilocks_core.steps import ExitStatusCheck, SharedContext
from goldilocks_core.system_settings import SystemSettings

_CTX = SharedContext(prefix="pwscf", outdir="./out", pseudo_dir="./pseudo")


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
        cell=structure,
        ordering=OrderingRecord(requested="fm", applied="fm"),
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
        # A neutral cell (z_valence=4.0 per atom below): no tot_charge
        # unless a test overrides electron_count itself.
        electron_count=ElectronCountDecision(
            nelec=4.0 * len(structure), tot_charge=0.0
        ),
        pseudopotentials=tuple(
            pseudo_metadata_factory(element, functional=functional, z_valence=4.0)
            for element in elements
        ),
        magnetic=magnetic or _magnetic(structure),
        vdw=vdw or VdwFacts(use_vdw=False, method=None),
        hubbard=hubbard or HubbardUDecision(),
        boundary=boundary or BoundaryFacts(isolation="periodic"),
    )


_UNSET = object()


def _step(
    *,
    occupations: OccupationsDecision | None = _UNSET,
    k_sampling: KSamplingDecision | None = _UNSET,
    nbnd: NbndDecision | None = _UNSET,
    convergence: ConvergenceDecision | None = _UNSET,
    parallel: ParallelisationDecision | None = _UNSET,
    max_seconds: int | None = 6946,
    relax=None,
) -> PwSettings:
    return PwSettings(
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
            ParallelisationDecision(npool=2, ndiag=None, kpoint_spin_pairs=4)
            if parallel is _UNSET
            else parallel
        ),
        max_seconds=max_seconds,
        relax=relax,
    )


def test_tot_charge_is_the_electron_count_decision_s_charge(
    silicon_structure, pseudo_metadata_factory
) -> None:
    """QE's &SYSTEM namelist has no ``nelec`` input keyword (ASE's
    ALL_KEYS['pw']['system'] lists ``tot_charge`` only, #34): a requested
    count reaches pw.x as the charge the electron-count decision derived,
    rendered verbatim."""
    system = _system(silicon_structure, pseudo_metadata_factory)
    system = dataclasses.replace(
        system, electron_count=ElectronCountDecision(nelec=3.0, tot_charge=1.0)
    )
    step = _step()

    content = write_qe_scf(system, step, _CTX)[0].files["scf.in"]

    assert "tot_charge       = 1.0" in content
    assert "nelec" not in content


@pytest.mark.parametrize("nelec", [4.0, 3.0, 8.0])
def test_a_neutral_decision_renders_no_tot_charge_whatever_the_count(
    silicon_structure, pseudo_metadata_factory, nelec: float
) -> None:
    """The writer never derives a charge from two electron counts (#128):
    an AFM cell counted on the uploaded cell turned the difference into
    tot_charge = 24. Only the decision's own charge is rendered, so a
    neutral decision renders none, however its count compares with the
    pseudopotentials' z_valence sum (4.0 here)."""
    system = _system(silicon_structure, pseudo_metadata_factory)
    system = dataclasses.replace(
        system, electron_count=ElectronCountDecision(nelec=nelec, tot_charge=0.0)
    )
    step = _step()

    content = write_qe_scf(system, step, _CTX)[0].files["scf.in"]

    assert "tot_charge" not in content


def test_writes_one_scf_step_with_pure_translated_content(
    silicon_structure, pseudo_metadata_factory
) -> None:
    system = _system(silicon_structure, pseudo_metadata_factory)
    step = _step()

    steps = write_qe_scf(system, step, _CTX)

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
    assert "max_seconds      = 6946" in content
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
    assert "disk_io" not in content


@pytest.mark.parametrize("purpose, seconds", [("scf", 6946), ("nscf", 6733)])
def test_max_seconds_is_the_steps_own_share_of_the_walltime(
    diamond_silicon, pseudo_metadata_factory, purpose, seconds
) -> None:
    """#169: pw.x counts max_seconds from its own start, so each step
    renders its own share (PwSettings.max_seconds), not the job's."""
    system = _system(diamond_silicon, pseudo_metadata_factory)

    (rendered,) = write_qe_scf(
        system, _step(max_seconds=seconds), _CTX, purpose=purpose
    )

    lines = rendered.files[f"{purpose}.in"].splitlines()
    assert [line for line in lines if "max_seconds" in line] == [
        f"   max_seconds      = {seconds}"
    ]


@pytest.mark.parametrize("purpose", ["scf", "nscf"])
def test_pw_x_step_is_checked_by_the_status_it_always_writes(
    diamond_silicon, pseudo_metadata_factory, purpose
) -> None:
    """#129: pw.x records every exit_status in <prefix>.save's schema XML;
    the <prefix>.xml copy is missing after a non-converged stop."""
    system = _system(diamond_silicon, pseudo_metadata_factory)

    (rendered,) = write_qe_scf(system, _step(), _CTX, purpose=purpose)

    assert rendered.completion == ExitStatusCheck(
        "./out/pwscf.save/data-file-schema.xml"
    )


def test_status_file_follows_the_shared_prefix_and_outdir(
    diamond_silicon, pseudo_metadata_factory
) -> None:
    system = _system(diamond_silicon, pseudo_metadata_factory)
    ctx = SharedContext(prefix="si", outdir="/scratch/run", pseudo_dir="./pseudo")

    (rendered,) = write_qe_scf(system, _step(), ctx)

    assert rendered.completion == ExitStatusCheck(
        "/scratch/run/si.save/data-file-schema.xml"
    )


def test_purpose_nscf_renames_the_step_and_the_calculation_keyword(
    silicon_structure, pseudo_metadata_factory
) -> None:
    """v2 epic 9 (#9): DOS's nscf step is the same writer, a different
    ``purpose`` -- QE's own documented ``calculation='nscf'`` value, not
    a new scientific decision this writer makes."""
    system = _system(silicon_structure, pseudo_metadata_factory)
    step = _step()

    steps = write_qe_scf(system, step, _CTX, purpose="nscf")

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
    step = _step(
        parallel=ParallelisationDecision(npool=4, ndiag=16, kpoint_spin_pairs=4)
    )

    steps = write_qe_scf(system, step, _CTX)

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

    content = write_qe_scf(system, step, _CTX)[0].files["scf.in"]

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

    content = write_qe_scf(system, _step(), _CTX)[0].files["scf.in"]

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
        write_qe_scf(system, _step(), _CTX)


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

    content = write_qe_scf(system, _step(), _CTX)[0].files["scf.in"]

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


@pytest.mark.parametrize(
    ("spin_polarized", "spin_orbit_enabled", "layout", "rendered", "absent"),
    [
        (False, False, (1, 1, 1), (), ("nspin", "noncolin", "starting_magn")),
        (True, False, (2, 1, 2), ("nspin            = 2",), ("noncolin",)),
        # noncolin makes pw.x run nspin=4 (QE 7.3 PW/src/input.f90:718),
        # with four magnetization components only when a starting
        # magnetization is set (input.f90:1589, set_spin_vars.f90:29-41)
        (
            False,
            True,
            (4, 2, 1),
            ("noncolin         = .true.",),
            ("nspin", "starting_magn"),
        ),
        (
            True,
            True,
            (4, 2, 4),
            ("noncolin         = .true.", "starting_magnetization(1)"),
            ("nspin",),
        ),
    ],
)
def test_qe_spin_layout_matches_the_spin_keywords_the_writer_renders(
    pseudo_metadata_factory,
    spin_polarized: bool,
    spin_orbit_enabled: bool,
    layout: tuple[int, int, int],
    rendered: tuple[str, ...],
    absent: tuple[str, ...],
) -> None:
    """``qe_spin_layout`` sizes pools (#168) and pw.x's memory (#148) for
    the run this writer describes, so the two must agree on every
    magnetic configuration."""
    feco = Structure(
        Lattice.cubic(2.857), ["Fe", "Co"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    )
    magnetic = _magnetic(
        feco,
        spin_polarized=spin_polarized,
        starting_magnetization={"Fe": 0.3, "Co": 0.2} if spin_polarized else None,
        spin_orbit_enabled=spin_orbit_enabled,
    )
    system = _system(feco, pseudo_metadata_factory, magnetic=magnetic)

    content = write_qe_scf(system, _step(), _CTX)[0].files["scf.in"]

    assert qe_spin_layout(magnetic) == layout
    for keyword in rendered:
        assert keyword in content
    for keyword in absent:
        assert keyword not in content


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

    content = write_qe_scf(system, _step(), _CTX)[0].files["scf.in"]

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


def test_non_afm_structure_collapses_source_labels_to_one_species_per_element(
    pseudo_metadata_factory,
) -> None:
    """Real CIF files often carry their own per-atom label per
    crystallographic site (e.g. ``Si1``..``Si40`` for a plain,
    non-magnetic silicon cell) that has nothing to do with magnetism --
    AFM species-splitting did not run here (`afm_relabeled=False`, the
    default), so those source labels must not leak into ATOMIC_SPECIES/
    ATOMIC_POSITIONS as if they were real, distinct QE species."""
    structure = Structure(
        Lattice.cubic(5.43),
        ["Si", "Si", "Si"],
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25], [0.5, 0.5, 0.0]],
        labels=["Si1", "Si7", "Si23"],
    )
    system = _system(structure, pseudo_metadata_factory)

    content = write_qe_scf(system, _step(), _CTX)[0].files["scf.in"]

    assert "ntyp             = 1" in content
    species = content.split("ATOMIC_SPECIES\n")[1].split("\n\n")[0]
    assert species == "  Si  28.0855  Si.UPF"
    positions = content.split("ATOMIC_POSITIONS")[1]
    assert "Si1" not in positions
    assert "Si7" not in positions
    assert "Si23" not in positions
    assert positions.count("Si  ") == 3


def test_vdw_method_translates_to_qe_keyword(
    silicon_structure, pseudo_metadata_factory
) -> None:
    system = _system(
        silicon_structure,
        pseudo_metadata_factory,
        vdw=VdwFacts(use_vdw=True, method="d3bj"),
    )

    content = write_qe_scf(system, _step(), _CTX)[0].files["scf.in"]

    assert "vdw_corr         = 'grimme-d3'" in content
    assert "dftd3_version    = 4" in content


def test_a_periodic_boundary_writes_tstress_and_no_assume_isolated(
    diamond_silicon, pseudo_metadata_factory
) -> None:
    system = _system(
        diamond_silicon,
        pseudo_metadata_factory,
        boundary=BoundaryFacts(isolation="periodic"),
    )

    content = write_qe_scf(system, _step(), _CTX)[0].files["scf.in"]

    assert "tstress          = .true." in content
    assert "assume_isolated" not in content


def test_a_slab_boundary_writes_the_2d_truncation_and_keeps_tstress(
    pseudo_metadata_factory,
) -> None:
    """#161: 'slab' is QE's 2D Coulomb truncation, which computes energies,
    forces and stresses in a two-dimensional framework (QE 7.3
    PW/Doc/INPUT_PW.def:1655-1675)."""
    system = _system(
        graphene_monolayer(),
        pseudo_metadata_factory,
        boundary=BoundaryFacts(isolation="slab"),
    )

    content = write_qe_scf(system, _step(), _CTX)[0].files["scf.in"]

    assert "assume_isolated  = '2D'" in content
    assert "tstress          = .true." in content


def test_an_isolated_boundary_writes_martyna_tuckerman_and_no_tstress(
    pseudo_metadata_factory,
) -> None:
    """#126: pw.x computes no stress for an isolated system and clears a
    requested tstress (QE 7.3 PW/src/input.f90:1361-1365), so the input
    does not ask for one."""
    system = _system(
        carbon_pair_in_a_box(),
        pseudo_metadata_factory,
        boundary=BoundaryFacts(isolation="isolated"),
    )

    content = write_qe_scf(system, _step(), _CTX)[0].files["scf.in"]

    assert "assume_isolated  = 'martyna-tuckerman'" in content
    assert "tstress" not in content
    assert "tprnfor          = .true." in content


def test_smearing_occupations_emit_smearing_type_and_degauss(
    silicon_structure, pseudo_metadata_factory
) -> None:
    system = _system(silicon_structure, pseudo_metadata_factory)
    step = _step(
        occupations=OccupationsDecision(
            occupations="smearing", smearing_type="cold", degauss=0.01
        )
    )

    content = write_qe_scf(system, step, _CTX)[0].files["scf.in"]

    assert "occupations      = 'smearing'" in content
    assert "smearing         = 'cold'" in content
    assert "degauss          = 0.01" in content


def _diamond_silicon() -> Structure:
    a = 5.43
    return Structure(
        Lattice([[0.0, a / 2, a / 2], [a / 2, 0.0, a / 2], [a / 2, a / 2, 0.0]]),
        ["Si", "Si"],
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )


def test_unset_mixing_fixed_ns_writes_no_keyword(pseudo_metadata_factory) -> None:
    """#127: unset means QE's own default of 0, so &ELECTRONS carries no
    mixing_fixed_ns line at all."""
    system = _system(_diamond_silicon(), pseudo_metadata_factory)

    content = write_qe_scf(system, _step(), _CTX)[0].files["scf.in"]

    assert "mixing_fixed_ns" not in content


def test_explicit_mixing_fixed_ns_is_written(pseudo_metadata_factory) -> None:
    system = _system(_diamond_silicon(), pseudo_metadata_factory)
    convergence = dataclasses.replace(_step().convergence, mixing_fixed_ns=10)
    step = _step(convergence=convergence)

    content = write_qe_scf(system, step, _CTX)[0].files["scf.in"]

    electrons = content.split("&ELECTRONS")[1].split("\n/\n")[0]
    assert "mixing_fixed_ns  = 10" in electrons


def _hubbard_card(content: str) -> list[str]:
    """The HUBBARD card's lines, header first; [] when there is none."""
    lines = content.splitlines()
    if "HUBBARD (ortho-atomic)" not in lines:
        return []
    start = lines.index("HUBBARD (ortho-atomic)")
    card = []
    for line in lines[start:]:
        if not line.strip():
            break
        card.append(line)
    return card


def _nickel_oxide() -> Structure:
    return Structure(
        Lattice.from_parameters(2.95, 2.95, 2.95, 60, 60, 60),
        ["Ni", "O"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )


def test_u_values_render_the_qe_hubbard_card(
    silicon_structure, pseudo_metadata_factory
) -> None:
    """QE >= 7.1's onsite U line is species-label-keyed plus its Hubbard
    manifold, not atom-index-keyed -- verified against QE 7.3's own
    Doc/Hubbard_input.tex ("U Mn-3d 5.0") and Modules/read_cards.f90's
    card_hubbard parsing (#88)."""
    system = _system(
        silicon_structure,
        pseudo_metadata_factory,
        hubbard=HubbardUDecision(u_values=(HubbardU("Si", "3p", 3.0, "human"),)),
    )

    content = write_qe_scf(system, _step(), _CTX)[0].files["scf.in"]

    assert _hubbard_card(content) == ["HUBBARD (ortho-atomic)", "  U  Si-3p  3"]


def test_no_u_values_render_no_hubbard_card(pseudo_metadata_factory) -> None:
    """#135: a correlated oxide whose elements get no U (plain DFT) renders
    no card at all -- QE 7.3 stops on a HUBBARD card without a positive
    value -- and the writer does not raise."""
    system = _system(_nickel_oxide(), pseudo_metadata_factory)

    content = write_qe_scf(system, _step(), _CTX)[0].files["scf.in"]

    assert "HUBBARD" not in content


def test_only_elements_with_a_u_value_get_a_hubbard_line(
    pseudo_metadata_factory,
) -> None:
    system = _system(
        _nickel_oxide(),
        pseudo_metadata_factory,
        hubbard=HubbardUDecision(u_values=(HubbardU("Ni", "3d", 6.2, "table"),)),
    )

    content = write_qe_scf(system, _step(), _CTX)[0].files["scf.in"]

    assert _hubbard_card(content) == ["HUBBARD (ortho-atomic)", "  U  Ni-3d  6.2"]


def test_each_species_of_a_split_element_gets_the_element_s_u_line(
    pseudo_metadata_factory,
) -> None:
    """Two Fe sites with opposite starting magnetization become two QE
    species, Fe1 and Fe2; each needs its own U line, keyed by the label
    ATOMIC_SPECIES actually uses (#88). Lines follow label order, not site
    order."""
    structure = Structure(
        Lattice.cubic(4.3),
        ["O", "O", "Fe", "Fe"],
        [[0.25, 0.25, 0.25], [0.75, 0.75, 0.75], [0, 0, 0], [0.5, 0.5, 0.5]],
        labels=["O", "O", "Fe_down", "Fe_up"],
    )
    magnetic = _magnetic(
        structure,
        spin_polarized=True,
        starting_magnetization={"Fe_up": 0.5, "Fe_down": -0.5},
    )
    system = _system(
        structure,
        pseudo_metadata_factory,
        magnetic=magnetic,
        hubbard=HubbardUDecision(
            u_values=(
                HubbardU("Fe", "3d", 5.3, "table"),
                HubbardU("O", "2p", 1.5, "human"),
            )
        ),
    )

    content = write_qe_scf(system, _step(), _CTX)[0].files["scf.in"]

    assert _hubbard_card(content) == [
        "HUBBARD (ortho-atomic)",
        "  U  Fe1-3d  5.3",
        "  U  Fe2-3d  5.3",
        "  U  O-2p  1.5",
    ]


def test_cif_site_labels_render_one_line_per_qe_species(
    pseudo_metadata_factory,
) -> None:
    """Regression (#88): a CIF-loaded structure numbers every site (V0,
    Cl1, Cl2); the card must name the species label ATOMIC_SPECIES uses,
    once, and never a stray label."""
    loaded = Structure.from_str(
        Structure(
            Lattice.cubic(4.0),
            ["V", "Cl", "Cl"],
            [[0, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0.5]],
        ).to(fmt="cif"),
        fmt="cif",
    )
    system = _system(
        loaded,
        pseudo_metadata_factory,
        hubbard=HubbardUDecision(u_values=(HubbardU("V", "3d", 3.25, "table"),)),
    )

    content = write_qe_scf(system, _step(), _CTX)[0].files["scf.in"]

    assert _hubbard_card(content) == ["HUBBARD (ortho-atomic)", "  U  V-3d  3.25"]
    assert "  V  " in content.split("ATOMIC_SPECIES")[1].split("\n\n")[0]


@pytest.mark.parametrize(
    "field",
    ["occupations", "k_sampling", "nbnd", "convergence", "parallel", "max_seconds"],
)
def test_missing_required_step_setting_raises_generation_error(
    silicon_structure, pseudo_metadata_factory, field
) -> None:
    system = _system(silicon_structure, pseudo_metadata_factory)
    step = _step(**{field: None})

    with pytest.raises(GenerationError, match=field):
        write_qe_scf(system, step, _CTX)


def test_relax_settings_on_an_scf_step_is_rejected(
    silicon_structure, pseudo_metadata_factory
) -> None:
    from goldilocks_core.advisors.relax import RelaxOptions

    system = _system(silicon_structure, pseudo_metadata_factory)
    step = _step(relax=RelaxOptions())

    with pytest.raises(GenerationError, match="relax"):
        write_qe_scf(system, step, _CTX)


def test_missing_pseudopotential_element_raises_generation_error(
    pseudo_metadata_factory,
) -> None:
    structure = Structure(
        Lattice.cubic(5.64), ["Na", "Cl"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    )
    system = SystemSettings(
        functional="PBEsol",
        cutoffs=CutoffsDecision(ecutwfc_ry=30.0, ecutrho_ry=120.0),
        electron_count=ElectronCountDecision(nelec=8.0, tot_charge=0.0),
        pseudopotentials=(pseudo_metadata_factory("Na"),),
        magnetic=_magnetic(structure),
        vdw=VdwFacts(use_vdw=False, method=None),
        hubbard=HubbardUDecision(),
        boundary=BoundaryFacts(isolation="periodic"),
    )

    with pytest.raises(GenerationError, match="missing"):
        write_qe_scf(system, _step(), _CTX)


def test_pseudopotential_functional_mismatch_raises_generation_error(
    silicon_structure, pseudo_metadata_factory
) -> None:
    system = SystemSettings(
        functional="PBEsol",
        cutoffs=CutoffsDecision(ecutwfc_ry=30.0, ecutrho_ry=120.0),
        electron_count=ElectronCountDecision(nelec=4.0, tot_charge=0.0),
        pseudopotentials=(pseudo_metadata_factory("Si", functional="LDA"),),
        magnetic=_magnetic(silicon_structure),
        vdw=VdwFacts(use_vdw=False, method=None),
        hubbard=HubbardUDecision(),
        boundary=BoundaryFacts(isolation="periodic"),
    )

    with pytest.raises(GenerationError, match="functional mismatch"):
        write_qe_scf(system, _step(), _CTX)


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
        electron_count=ElectronCountDecision(nelec=4.0, tot_charge=0.0),
        pseudopotentials=(pseudo_metadata_factory("Si"), pseudo_metadata_factory("Ge")),
        magnetic=magnetic,
        vdw=VdwFacts(use_vdw=False, method=None),
        hubbard=HubbardUDecision(),
        boundary=BoundaryFacts(isolation="periodic"),
    )

    with pytest.raises(GenerationError, match="disordered"):
        write_qe_scf(system, _step(), _CTX)
