from __future__ import annotations

import pytest
from pymatgen.core import Lattice, Structure

from goldilocks_core.advisors.k_sampling import KSamplingDecision
from goldilocks_core.advisors.magnetic_config import (
    MagneticCell,
    MagneticConfigHumanInput,
    magnetic_cell,
)
from goldilocks_core.advisors.n_irr_k import NIrrKHumanInput, n_irr_k
from goldilocks_core.resolution import Blocked, Provenance, Resolved, Unavailable

_DIAMOND_SILICON = Structure.from_spacegroup(
    "Fd-3m", Lattice.cubic(5.43), ["Si"], [[0.0, 0.0, 0.0]]
)
_GRAPHITE = Structure.from_spacegroup(
    "P6_3/mmc",
    Lattice.hexagonal(2.46, 6.71),
    ["C", "C"],
    [[0.0, 0.0, 0.25], [1 / 3, 2 / 3, 0.25]],
)
_RUTILE = Structure.from_spacegroup(
    "P4_2/mnm",
    Lattice.tetragonal(4.594, 2.959),
    ["Ti", "O"],
    [[0.0, 0.0, 0.0], [0.305, 0.305, 0.0]],
)


_FECO = Structure(Lattice.cubic(2.80), ["Fe", "Co"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def _k_sampling(
    mesh: tuple[int, int, int], shift: tuple[int, int, int] = (0, 0, 0)
) -> Resolved[KSamplingDecision]:
    return Resolved(
        KSamplingDecision(mesh=mesh, shift=shift, k_distance=0.2),
        Provenance(source="heuristic"),
    )


def _cell(
    structure: Structure, magnetism: str = "non_magnetic", **settings: object
) -> MagneticCell:
    """The magnetic step's cell for ``structure``: the one n_irr_k counts."""
    return magnetic_cell(
        structure,
        Resolved(magnetism, Provenance(source="heuristic")),
        MagneticConfigHumanInput(**settings),
    ).value


@pytest.mark.parametrize(
    ("structure", "mesh", "irreducible"),
    [
        (_DIAMOND_SILICON, (4, 4, 4), 10),  # 64 points, cubic
        (_DIAMOND_SILICON, (5, 5, 5), 10),  # 125 points
        (_GRAPHITE, (14, 14, 5), 72),  # 980 points, hexagonal
        (_RUTILE, (7, 7, 10), 60),  # 490 points, tetragonal
    ],
)
def test_reduces_a_mesh_by_symmetry(
    structure: Structure, mesh: tuple[int, int, int], irreducible: int
) -> None:
    """Real crystals reduce by a large factor. A fallback to the full mesh
    size would report an ordinary-looking integer, which is why there is
    none. n_irr_k is the only place a mesh is reduced; the ladder carries
    no counts (#155)."""
    state = n_irr_k(_cell(structure), _k_sampling(mesh))

    assert state.ok
    assert state.value == irreducible


def test_gamma_only_mesh_has_exactly_one_irreducible_kpoint() -> None:
    state = n_irr_k(_cell(_DIAMOND_SILICON), _k_sampling((1, 1, 1)))

    assert state.value == 1


def test_nosym_returns_the_full_mesh_size_unreduced() -> None:
    state = n_irr_k(
        _cell(_DIAMOND_SILICON),
        _k_sampling((4, 4, 4)),
        human=NIrrKHumanInput(nosym=True),
    )

    assert state.value == 64
    assert state.source == "human"


def test_blocked_k_sampling_propagates_as_blocked() -> None:
    state = n_irr_k(_cell(_DIAMOND_SILICON), Blocked(by="synthetic failure"))

    assert not state.ok
    assert state.status == "blocked"
    assert state.root_cause() == "synthetic failure"


def test_unavailable_k_sampling_is_treated_as_blocked_not_a_default() -> None:
    """There is no safe made-up mesh to fall back on if k_sampling itself
    could not resolve -- this must block, not guess."""
    state = n_irr_k(_cell(_DIAMOND_SILICON), Unavailable(reason="synthetic failure"))

    assert not state.ok
    assert state.status == "blocked"
    assert state.root_cause() == "synthetic failure"


def test_counts_a_spin_split_cell_by_its_species_not_its_elements() -> None:
    """#170: pw.x reduces k-points by the operations that map each species
    onto itself, and the species of a spin-polarized run follow its
    starting magnetization. A layered split of conventional fcc Ni (z = 0
    up, z = 1/2 down) keeps 4/mmm, whose 16 operations leave 40 orbits of
    the 6x6x6 mesh, not m-3m's 20 for the bare element: the cubic count
    sized pools and memory for too few k-points."""
    fcc_nickel = Structure(
        Lattice.cubic(3.52),
        ["Ni"] * 4,
        [[0, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5]],
        labels=["Ni0", "Ni1", "Ni2", "Ni3"],
    )
    split = {"Ni0": 0.5, "Ni1": 0.5, "Ni2": -0.5, "Ni3": -0.5}
    cell = _cell(fcc_nickel, "magnetic", starting_magnetization=split)
    mesh = _k_sampling((6, 6, 6))

    assert n_irr_k(cell, mesh).value == 40
    assert n_irr_k(_cell(fcc_nickel), mesh).value == 20


def test_counts_the_fept_candidate_pw_x_ran_as_pw_x_did() -> None:
    """#170: the FePt (MC3D mc3d-49510) deck pw.x ran as the listing's
    afm-4 before this change, its Pt split into Pt1/Pt2 like its Fe, on an
    8x7x7 mesh: pw.x 7.5 found 8 symmetry operations and 80 k-points per
    spin (SCARF job 3654948). Counting the bare elements gave 50, and npool
    was sized from it. The listing now calls that cell afm-2 and leaves its
    Pt unspun, one species."""
    fept = Structure(
        Lattice.orthorhombic(3.72103604, 3.79793688, 3.79793688),
        ["Fe", "Fe", "Pt", "Pt"],
        [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5]],
        labels=["Fe1", "Fe2", "Pt1", "Pt2"],
    )
    split = {"Fe1": 0.1, "Fe2": -0.1, "Pt1": 0.1, "Pt2": -0.1}
    cell = _cell(fept, "magnetic", starting_magnetization=split)
    mesh = _k_sampling((8, 7, 7))

    assert [site.label for site in cell.structure] == ["Fe1", "Fe2", "Pt1", "Pt2"]
    assert n_irr_k(cell, mesh).value == 80
    assert n_irr_k(_cell(fept), mesh).value == 50


def test_spin_orbit_without_moments_keeps_the_structural_count() -> None:
    """#172: pw.x drops time-reversal pairing only when the noncollinear
    run is magnetic (``magnetic_sym = noncolin .AND. domag``, QE 7.3
    PW/src/setup.f90:484-485). Spin-orbit coupling on a non-magnetic run
    counts as the structure does."""
    mesh = _k_sampling((4, 4, 4))

    spin_orbit = n_irr_k(_cell(_DIAMOND_SILICON, spin_orbit_coupling=True), mesh)

    assert spin_orbit.value == n_irr_k(_cell(_DIAMOND_SILICON), mesh).value == 10


def test_collinear_moments_keep_the_structural_count() -> None:
    """#172: a collinear run's moments have no direction pw.x checks, so
    the count is the species' structural one (56 for B2 FeCo on 10^3),
    whatever moment_direction says."""
    collinear = _cell(_FECO, "magnetic", moment_direction=(0, 0, 1))

    assert n_irr_k(collinear, _k_sampling((10, 10, 10))).value == 56


@pytest.mark.parametrize(
    ("direction", "irreducible"),
    [
        ((0, 0, 1), 126),  # 4/mm'm': k-space point group 4/mmm, 16 operations
        ((1, 1, 1), 116),  # -3m': k-space point group -3m, 12 operations
        ((1, 1, 0), 186),  # m'm'm: k-space point group mmm, 8 operations
    ],
)
def test_noncollinear_moments_count_by_their_direction(
    direction: tuple[int, int, int], irreducible: int
) -> None:
    """#172: the moment direction decides which operations of cubic B2
    FeCo survive (pymatgen's cubic cell, so [uvw] is Cartesian here)."""
    cell = _cell(
        _FECO, "magnetic", spin_orbit_coupling=True, moment_direction=direction
    )

    assert n_irr_k(cell, _k_sampling((10, 10, 10))).value == irreducible


def test_noncollinear_moments_count_a_shifted_mesh_with_its_shift() -> None:
    """#172: the magnetic count keeps the mesh's origin offset."""
    cell = _cell(_FECO, "magnetic", spin_orbit_coupling=True)

    gamma = n_irr_k(cell, _k_sampling((4, 4, 4))).value
    shifted = n_irr_k(cell, _k_sampling((4, 4, 4), shift=(1, 1, 1))).value

    assert (gamma, shifted) == (18, 6)


def test_noncollinear_moments_keep_k_and_minus_k_apart_without_inversion() -> None:
    """#172: pw.x drops the plain k -> -k pairing for a magnetic
    noncollinear run; with inversion symmetry the kept operations pair k
    with -k anyway, without it they do not. In zincblende beta-MnS (no
    inversion) with moments along c, the operations kept give -42m on k:
    52 points of a 6^3 mesh, where pairing k with -k would give 40."""
    zincblende = Structure.from_spacegroup(
        "F-43m", Lattice.cubic(5.60), ["Mn", "S"], [[0, 0, 0], [0.25, 0.25, 0.25]]
    )
    cell = _cell(zincblende, "magnetic", spin_orbit_coupling=True)

    assert n_irr_k(cell, _k_sampling((6, 6, 6))).value == 52


def test_spin_orbit_with_every_starting_moment_zero_counts_as_non_magnetic() -> None:
    """#172: pw.x reduces by magnetic symmetry only when some species
    starts with a moment (``domag``, QE 7.3 PW/src/setup.f90:214); zero
    starting moments everywhere keep the structural count."""
    cell = _cell(
        _FECO,
        "magnetic",
        spin_orbit_coupling=True,
        starting_magnetization={"Fe": 0.0, "Co": 0.0},
    )

    assert n_irr_k(cell, _k_sampling((10, 10, 10))).value == 56
