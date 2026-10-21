"""The k-mesh ladder, checked on real materials -- ported from
goldilocks-data's own test_kmesh.py (v2 epic 6, #6) so the rungs this
module builds (kindex, mesh and intervals, which depend on the lattice
alone) match goldilocks-data's on the same lattices, per the epic's
acceptance criteria. Core's rungs carry
no symmetry-reduced count (#155); test_advisors_n_irr_k.py checks the
reduction. Every structure here is a real crystal: three built from their
space group and published lattice constants, and two MC3D cells --
hexagonal Si exactly as goldilocks-data's own SCF campaign ran it
(embedded inline rather than as a separate fixture file, since it is used
by exactly one test) and triclinic TiNi. Synthetic lattices were tried
first upstream and hid what a real cell exposes: the repeated-mesh skip
never fired.
"""

from __future__ import annotations

import math
from dataclasses import astuple
from itertools import pairwise

import pytest
from pymatgen.core import Lattice, Structure

from goldilocks_core.kmesh import (
    build_gamma_kmesh_entries,
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
    """Bernal graphite, C on the 2b and 2c sites of P6_3/mmc, a = 2.46 A,
    c = 6.71 A. Layered: |b_1| is 3.1x |b_3|. goldilocks-data's fixture
    places only the 2b site, a simple-hexagonal cell on the same lattice."""
    return Structure.from_spacegroup(
        "P6_3/mmc",
        Lattice.hexagonal(2.46, 6.71),
        ["C", "C"],
        [[0.0, 0.0, 0.25], [1 / 3, 2 / 3, 0.25]],
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


def mc3d_31340_tini() -> Structure:
    """Triclinic TiNi, MC3D 31340 (P-1). Its three reciprocal axes all
    differ, and its Laue group, -1, maps every mesh onto itself."""
    return Structure(
        Lattice.from_parameters(
            2.686185331044557,
            6.97034299971133,
            4.530142290302962,
            110.41779674117039,
            90.51114535957441,
            100.25528065043186,
        ),
        ["Ti", "Ti", "Ti", "Ni", "Ni", "Ni"],
        [
            [0.1585096067374422, 0.32623940974867155, 0.7676125146233795],
            [0.8414903932281635, 0.6737605902653331, 0.23238748538371365],
            [0.0, 0.0, 0.0],
            [0.3408662647191799, 0.6631612739114513, 0.7064985079217144],
            [0.6591337352464222, 0.336838726102553, 0.293501492085372],
            [0.4999999999814244, 0.9999999999993829, 0.5000000000107764],
        ],
    )


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


_TINI_LADDER = (
    # kindex, mesh, k_distance_interval, k_line_density_interval, k_pra
    (1, (1, 1, 1), (2.38413115, math.inf), (3.20461063, 3.95312898), 6.0),
    (2, (2, 1, 1), (1.48436263, 2.38413115), (3.95312898, 6.349377), 12.0),
    (3, (2, 1, 2), (1.19206558, 1.48436263), (6.349377, 6.5885483), 24.0),
    (4, (3, 1, 2), (0.98033522, 1.19206558), (6.5885483, 9.22396762), 36.0),
    (5, (3, 2, 2), (0.79471038, 0.98033522), None, 72.0),
    (6, (4, 2, 2), (0.74218132, 0.79471038), (9.61383189, 10.582295), 96.0),
    (7, (4, 2, 3), (0.59603279, 0.74218132), (10.582295, 11.85938694), 144.0),
    (8, (5, 2, 3), (0.49478754, 0.59603279), (11.85938694, 14.49480626), 180.0),
    (9, (5, 2, 4), (0.49016761, 0.49478754), None, 240.0),
    (10, (5, 3, 4), (0.47682623, 0.49016761), None, 360.0),
    (11, (6, 3, 4), (0.39735519, 0.47682623), (16.02305315, 17.13022558), 432.0),
    (12, (7, 3, 4), (0.37109066, 0.39735519), (17.13022558, 19.048131), 504.0),
    (13, (7, 3, 5), (0.34059016, 0.37109066), (19.048131, 19.7656449), 630.0),
    (14, (8, 3, 5), (0.32677841, 0.34059016), (19.7656449, 22.40106422), 720.0),
    (15, (8, 4, 5), (0.29801639, 0.32677841), None, 960.0),
    (16, (9, 4, 5), (0.29687253, 0.29801639), (22.43227441, 23.28104901), 1080.0),
    (17, (9, 4, 6), (0.26490346, 0.29687253), (23.28104901, 25.03648354), 1296.0),
    (18, (10, 4, 6), (0.24739377, 0.26490346), (25.03648354, 27.51396701), 1440.0),
    (19, (10, 4, 7), (0.2450838, 0.24739377), (27.51396701, 27.67190286), 1680.0),
    (20, (10, 5, 7), (0.23841312, 0.2450838), None, 2100.0),
    (21, (11, 5, 7), (0.2167392, 0.23841312), (28.84149566, 30.30732218), 2310.0),
    (22, (12, 5, 7), (0.2120518, 0.2167392), (30.30732218, 31.74688501), 2520.0),
)
_GRAPHITE_LADDER = (
    # kindex, mesh, k_distance_interval, k_line_density_interval, k_pra
    (1, (1, 1, 1), (2.94926726, math.inf), None, 4.0),
    (2, (2, 2, 1), (1.47463363, 2.94926726), (3.355, 5.32605623), 16.0),
    (3, (3, 3, 1), (0.98308909, 1.47463363), (5.32605623, 7.45647873), 36.0),
    (4, (4, 4, 1), (0.93639125, 0.98308909), (7.45647873, 9.58690122), 64.0),
    (5, (4, 4, 2), (0.73731681, 0.93639125), None, 128.0),
    (6, (5, 5, 2), (0.58985345, 0.73731681), (10.065, 11.71732371), 200.0),
    (7, (6, 6, 2), (0.49154454, 0.58985345), (11.71732371, 13.84774621), 288.0),
    (8, (7, 7, 2), (0.46819563, 0.49154454), (13.84774621, 15.9781687), 392.0),
    (9, (7, 7, 3), (0.42132389, 0.46819563), None, 588.0),
    (10, (8, 8, 3), (0.36865841, 0.42132389), (16.775, 18.10859119), 768.0),
    (11, (9, 9, 3), (0.32769636, 0.36865841), (18.10859119, 20.23901369), 972.0),
    (12, (10, 10, 3), (0.31213042, 0.32769636), (20.23901369, 22.36943618), 1200.0),
    (13, (10, 10, 4), (0.29492673, 0.31213042), None, 1600.0),
    (14, (11, 11, 4), (0.26811521, 0.29492673), (23.485, 24.49985867), 1936.0),
    (15, (12, 12, 4), (0.24577227, 0.26811521), (24.49985867, 26.63028117), 2304.0),
    (16, (13, 13, 4), (0.23409781, 0.24577227), (26.63028117, 28.76070366), 2704.0),
    (17, (13, 13, 5), (0.22686671, 0.23409781), None, 3380.0),
    (18, (14, 14, 5), (0.21066195, 0.22686671), (30.195, 30.89112615), 3920.0),
)
_DIAMOND_SILICON_LADDER = (
    # kindex, mesh, k_distance_interval, k_line_density_interval, k_pra
    (1, (1, 1, 1), (1.15712437, math.inf), (2.715, 8.145), 8.0),
    (2, (2, 2, 2), (0.57856218, 1.15712437), (8.145, 13.575), 64.0),
    (3, (3, 3, 3), (0.38570812, 0.57856218), (13.575, 19.005), 216.0),
    (4, (4, 4, 4), (0.28928109, 0.38570812), (19.005, 24.435), 512.0),
    (5, (5, 5, 5), (0.23142487, 0.28928109), (24.435, 29.865), 1000.0),
)


@pytest.mark.parametrize(
    ("structure", "reference"),
    [
        (mc3d_31340_tini(), _TINI_LADDER),
        (graphite(), _GRAPHITE_LADDER),
        (diamond_silicon(), _DIAMOND_SILICON_LADDER),
    ],
    ids=["triclinic TiNi", "hexagonal graphite", "cubic silicon"],
)
def test_ladder_matches_its_stored_reference(
    structure: Structure, reference: tuple[tuple[object, ...], ...]
) -> None:
    """Every field of every rung down to a 0.2 A^-1 floor, as the ladder
    stood at d254510. ``kindex`` is recorded in published records, so any
    change here renumbers them. The k-line-density bounds are stored to 8
    decimals; every other value is exact."""
    entries = build_gamma_kmesh_entries(structure, min_k_distance=0.2)

    assert [astuple(entry) for entry in entries] == [
        (kindex, mesh, k_distance, pytest.approx(k_line_density, abs=1e-8), k_pra)
        for kindex, mesh, k_distance, k_line_density, k_pra in reference
    ]
