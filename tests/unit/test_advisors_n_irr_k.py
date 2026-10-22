from __future__ import annotations

import pytest
from pymatgen.core import Lattice, Structure

from goldilocks_core.advisors.k_sampling import KSamplingDecision
from goldilocks_core.advisors.magnetic_config import (
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


def _k_sampling(mesh: tuple[int, int, int]) -> Resolved[KSamplingDecision]:
    return Resolved(
        KSamplingDecision(mesh=mesh, shift=(0, 0, 0), k_distance=0.2),
        Provenance(source="heuristic"),
    )


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
    state = n_irr_k(structure, _k_sampling(mesh))

    assert state.ok
    assert state.value == irreducible


def test_gamma_only_mesh_has_exactly_one_irreducible_kpoint() -> None:
    state = n_irr_k(_DIAMOND_SILICON, _k_sampling((1, 1, 1)))

    assert state.value == 1


def test_nosym_returns_the_full_mesh_size_unreduced() -> None:
    state = n_irr_k(
        _DIAMOND_SILICON, _k_sampling((4, 4, 4)), human=NIrrKHumanInput(nosym=True)
    )

    assert state.value == 64
    assert state.source == "human"


def test_blocked_k_sampling_propagates_as_blocked() -> None:
    state = n_irr_k(_DIAMOND_SILICON, Blocked(by="synthetic failure"))

    assert not state.ok
    assert state.status == "blocked"
    assert state.root_cause() == "synthetic failure"


def test_unavailable_k_sampling_is_treated_as_blocked_not_a_default() -> None:
    """There is no safe made-up mesh to fall back on if k_sampling itself
    could not resolve -- this must block, not guess."""
    state = n_irr_k(_DIAMOND_SILICON, Unavailable(reason="synthetic failure"))

    assert not state.ok
    assert state.status == "blocked"
    assert state.root_cause() == "synthetic failure"


def _split(structure: Structure, starting_magnetization: dict[str, float]) -> Structure:
    """The magnetic step's cell for ``structure``, its species split by the
    ``starting_magnetization`` of each site label."""
    return magnetic_cell(
        structure,
        Resolved("magnetic", Provenance(source="heuristic")),
        MagneticConfigHumanInput(starting_magnetization=starting_magnetization),
    ).value.structure


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
    cell = _split(fcc_nickel, {"Ni0": 0.5, "Ni1": 0.5, "Ni2": -0.5, "Ni3": -0.5})
    mesh = _k_sampling((6, 6, 6))

    assert n_irr_k(cell, mesh).value == 40
    assert n_irr_k(fcc_nickel, mesh).value == 20


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
    cell = _split(fept, {"Fe1": 0.1, "Fe2": -0.1, "Pt1": 0.1, "Pt2": -0.1})
    mesh = _k_sampling((8, 7, 7))

    assert [site.label for site in cell] == ["Fe1", "Fe2", "Pt1", "Pt2"]
    assert n_irr_k(cell, mesh).value == 80
    assert n_irr_k(fept, mesh).value == 50
