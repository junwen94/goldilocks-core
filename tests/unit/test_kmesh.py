"""The k-mesh ladder, checked on real materials -- ported from
goldilocks-data's own test_kmesh.py (v2 epic 6, #6) so the ladder this
module builds matches goldilocks-data's bit-for-bit on the same
structures, per the epic's acceptance criteria. Every structure here is a
real crystal: three built from their space group and published lattice
constants, and one from the exact MC3D cell goldilocks-data's own SCF
campaign ran (embedded inline rather than as a separate fixture file,
since it is used by exactly one test). Synthetic lattices were tried
first upstream and hid two things a real cell exposes -- symmetry
reduction never ran at all, and the repeated-mesh skip never fired.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import pytest
from pymatgen.core import Lattice, Structure

from goldilocks_core.kmesh import (
    build_gamma_kmesh_entries,
    entry_payload,
    generate_candidate_k_distances,
    k_distance_to_mesh,
)

_MC3D_67775_SILICON_CIF = """\
data_0

loop_
  _atom_site_label
  _atom_site_fract_x
  _atom_site_fract_y
  _atom_site_fract_z
  _atom_site_type_symbol
         Si1  0.6666666666716649  0.3333333333433297  0.06292314380102376  Si
         Si2  0.33333333332833526  0.6666666666566704  0.5629231437931663  Si
         Si3  0.6666666666716649  0.3333333333433297  0.4370768562068337  Si
         Si4  0.33333333332833526  0.6666666666566704  0.9370768561989763  Si
_cell_angle_alpha                       90.0
_cell_angle_beta                        90.0
_cell_angle_gamma                       120.00000000021976
_cell_length_a                          3.8504103266
_cell_length_b                          3.85041032657442
_cell_length_c                          6.3634082249
loop_
  _symmetry_equiv_pos_as_xyz
         'x, y, z'
_symmetry_int_tables_number             1
_symmetry_space_group_name_H-M          'P 1'
"""


def diamond_silicon() -> Structure:
    """Si, Fd-3m, a = 5.43 A. Cubic: 48 point-group operations, |b_i| all
    equal."""
    return Structure.from_spacegroup(
        "Fd-3m", Lattice.cubic(5.43), ["Si"], [[0.0, 0.0, 0.0]]
    )


def graphite() -> Structure:
    """C, P6_3/mmc, a = 2.46 A, c = 6.71 A. Layered: |b_1| is 3.1x |b_3|."""
    return Structure.from_spacegroup(
        "P6_3/mmc", Lattice.hexagonal(2.46, 6.71), ["C"], [[0.0, 0.0, 0.25]]
    )


def rutile() -> Structure:
    """TiO2, P4_2/mnm, a = 4.594 A, c = 2.959 A. Tetragonal: the short axis
    is c."""
    return Structure.from_spacegroup(
        "P4_2/mnm",
        Lattice.tetragonal(4.594, 2.959),
        ["Ti", "O"],
        [[0.0, 0.0, 0.0], [0.305, 0.305, 0.0]],
    )


def mc3d_67775_silicon() -> Structure:
    """Hexagonal Si, MC3D 67775, as goldilocks-data's SCF campaign ran it.

    Its a and b differ in the last few bits of the cell file
    (3.8504103266 against 3.85041032657442), which is what makes it
    interesting: see the repeated-mesh test.
    """
    return Structure.from_str(_MC3D_67775_SILICON_CIF, fmt="cif")


def test_gamma_kmesh_entries_start_with_gamma_mesh() -> None:
    entries = build_gamma_kmesh_entries(diamond_silicon(), min_k_distance=0.25)

    assert entries[0].kindex == 1
    assert entries[0].mesh == (1, 1, 1)
    assert entries[1].kindex == 2
    assert entries[0].k_pra == 8.0  # 8 sites in the conventional cell x 1


def test_k_distance_to_mesh_uses_solid_state_reciprocal_lengths() -> None:
    # Rutile: |b| = (1.3677, 1.3677, 2.1234) 1/A on the 2*pi reciprocal
    # lattice. ceil(|b_i| / 0.7) is (2, 2, 4). Reading the crystallographic
    # lengths instead -- the ones without 2*pi -- would give (1, 1, 1).
    assert k_distance_to_mesh(rutile(), 0.7) == (2, 2, 4)


def test_entry_payload_serializes_infinite_right_bound_as_none() -> None:
    payload = entry_payload(
        build_gamma_kmesh_entries(diamond_silicon(), min_k_distance=0.5)[0]
    )

    assert payload["kindex"] == 1
    assert payload["k_mesh"] == (1, 1, 1)
    assert payload["k_dist_right"] is None


def test_ladder_is_gap_free_down_to_the_floor() -> None:
    # Graphite is layered, so |b_1| = |b_2| = 2.949 against |b_3| = 0.936.
    # Under a per-axis k-point cap the two long axes would run out of
    # change points while the short one was still stepping, leaving holes.
    # A k-distance floor stops every axis at the same place, so the ladder
    # is gap-free by construction.
    meshes = [
        entry.mesh
        for entry in build_gamma_kmesh_entries(graphite(), min_k_distance=0.1)
    ]

    assert len(meshes) > 1
    for before, after in pairwise(meshes):
        steps = [now - previous for previous, now in zip(before, after, strict=True)]
        assert max(steps) == 1, f"{before} -> {after} skips a mesh"
        assert min(steps) >= 0, f"{before} -> {after} is not monotonic"


def test_ladder_drops_a_mesh_a_second_interval_would_repeat() -> None:
    # MC3D 67775, hexagonal Si. Its a and b are equal only to about 11
    # decimal places, so |b_1| and |b_2| give two change points a hair
    # apart, and the sliver of k-distance between them yields a mesh the
    # next interval yields again. Without the skip two kindex values would
    # name one mesh.
    #
    # Exactly equal axes do not cause this -- their change points coincide
    # and collapse in the candidate set. It takes a near miss, which is
    # why this needs a real cell: 36 of the 20,826 MC3D structures in
    # goldilocks-data's campaign hit it, and no idealised lattice does.
    structure = mc3d_67775_silicon()

    candidates = generate_candidate_k_distances(structure)
    entries = build_gamma_kmesh_entries(structure)

    # One interval per candidate, so fewer rungs than candidates means the
    # skip ran.
    assert len(entries) < len(candidates)
    assert len({entry.mesh for entry in entries}) == len(entries)


def test_lowering_the_floor_only_extends_the_ladder() -> None:
    # kindex is recorded in campaign snapshots and published records, so a
    # lower floor must never renumber a rung that already existed.
    structure = graphite()
    coarse = [
        entry.mesh for entry in build_gamma_kmesh_entries(structure, min_k_distance=0.2)
    ]
    fine = [
        entry.mesh for entry in build_gamma_kmesh_entries(structure, min_k_distance=0.1)
    ]

    assert len(fine) > len(coarse)
    assert fine[: len(coarse)] == coarse


def test_ladder_stops_at_the_resolution_floor() -> None:
    # The floor caps the densest mesh at ceil(|b_i| / min_k_distance) per
    # axis, independent of any k-point count. Taking the floor as |b| / 8
    # of a cubic cell makes that exactly (8, 8, 8) with no float noise.
    silicon = diamond_silicon()
    entries = build_gamma_kmesh_entries(
        silicon, min_k_distance=silicon.lattice.reciprocal_lattice.a / 8
    )

    assert entries[-1].mesh == (8, 8, 8)


def test_kindex_is_contiguous_and_one_based() -> None:
    entries = build_gamma_kmesh_entries(mc3d_67775_silicon())

    assert [entry.kindex for entry in entries] == list(range(1, len(entries) + 1))


def test_n_reduced_kpoints_is_the_irreducible_count_not_the_full_mesh() -> None:
    # Diamond silicon, the strongest symmetry there is. A fallback to the
    # full mesh size would report 64 here, and 64 is a perfectly
    # ordinary-looking integer -- which is the whole reason there is no
    # fallback.
    entries = build_gamma_kmesh_entries(diamond_silicon(), min_k_distance=0.2)
    by_mesh = {entry.mesh: entry for entry in entries}

    assert by_mesh[(4, 4, 4)].n_reduced_kpoints == 10
    assert by_mesh[(4, 4, 4)].k_pra == 8 * 64
    assert by_mesh[(1, 1, 1)].n_reduced_kpoints == 1


def test_every_real_crystal_reduces_by_a_large_factor() -> None:
    # Three real crystals, each at the densest rung of its own ladder.
    # Under the old fallback all three would have reported the full mesh
    # size, and the fallback was what the suite silently ran on -- so
    # nothing here would have failed. Pin the counts so that cannot recur.
    def densest(structure: Structure) -> tuple[int, int]:
        entry = build_gamma_kmesh_entries(structure, min_k_distance=0.2)[-1]
        full = entry.mesh[0] * entry.mesh[1] * entry.mesh[2]
        return full, entry.n_reduced_kpoints

    assert densest(diamond_silicon()) == (125, 10)  # (5, 5, 5), cubic
    assert densest(graphite()) == (980, 72)  # (14, 14, 5), hexagonal
    assert densest(rutile()) == (490, 60)  # (7, 7, 10), tetragonal


def test_an_unreducible_structure_raises_instead_of_counting_the_full_mesh() -> None:
    # The reduced count and the full mesh size are both ordinary integers,
    # so a caller cannot tell a fallback from a real answer.
    # goldilocks-core sizes memory and picks npool from this number: it
    # must fail loudly.
    @dataclass(frozen=True, slots=True)
    class Reciprocal:
        a: float
        b: float
        c: float

    @dataclass(frozen=True, slots=True)
    class FakeLattice:
        reciprocal_lattice: Reciprocal
        reciprocal_lattice_crystallographic: Reciprocal

    @dataclass(frozen=True, slots=True)
    class FakeStructure:
        lattice: FakeLattice

        def __len__(self) -> int:
            return 1

    fake = FakeStructure(
        lattice=FakeLattice(
            reciprocal_lattice=Reciprocal(1.0, 1.0, 1.0),
            reciprocal_lattice_crystallographic=Reciprocal(0.2, 0.2, 0.2),
        )
    )

    with pytest.raises(AttributeError):
        build_gamma_kmesh_entries(fake, min_k_distance=0.5)
