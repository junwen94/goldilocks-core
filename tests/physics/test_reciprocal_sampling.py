from __future__ import annotations

import itertools
import math

import pytest
from pymatgen.core import Lattice, Structure

from goldilocks_core.legacy_kmesh.math import (
    build_kmesh_entries,
    generate_candidate_k_distances,
    k_distance_to_mesh,
)


@pytest.mark.parametrize(
    ("lengths", "spacing"),
    [
        ((3.0, 4.0, 6.0), 0.4),
        ((2.5, 7.0, 11.0), 0.27),
        ((8.0, 8.0, 8.0), 0.15),
    ],
)
def test_mesh_is_minimal_grid_that_bounds_solid_state_reciprocal_spacing(
    lengths: tuple[float, float, float],
    spacing: float,
) -> None:
    structure = Structure(
        Lattice.orthorhombic(*lengths),
        ["Si"],
        [[0.0, 0.0, 0.0]],
    )

    mesh = k_distance_to_mesh(structure, spacing)
    reciprocal_lengths = tuple(2.0 * math.pi / length for length in lengths)
    expected = tuple(math.ceil(length / spacing) for length in reciprocal_lengths)

    assert mesh == expected
    for reciprocal_length, points in zip(reciprocal_lengths, mesh, strict=True):
        assert reciprocal_length / points <= spacing
        if points > 1:
            assert reciprocal_length / (points - 1) > spacing


def test_vacuum_axis_never_requests_zero_k_points() -> None:
    slab = Structure(
        Lattice.orthorhombic(2.46, 2.46, 40.0),
        ["C", "C"],
        [[0.0, 0.0, 0.5], [0.5, 0.5, 0.5]],
    )

    mesh = k_distance_to_mesh(slab, 0.2)

    assert mesh[:2] == (13, 13)
    assert mesh[2] == 1


def test_reciprocal_lattice_length_includes_the_2pi_factor() -> None:
    """A3: k-point spacing is only meaningful under one 2*pi convention. This
    pins the specific pymatgen behavior k_distance_to_mesh's docstring assumes
    (``Lattice.reciprocal_lattice`` already includes the 2*pi factor, the VASP
    KSPACING convention) — if a future pymatgen release changed this, every
    mesh k_distance_to_mesh produces would silently be wrong by a factor of
    2*pi, with no error anywhere."""
    lattice = Lattice.cubic(4.0)

    assert lattice.reciprocal_lattice.a == pytest.approx(2.0 * math.pi / 4.0)


def test_kmesh_ladder_never_has_an_axis_count_jump_larger_than_one() -> None:
    """A4: once the longest axis (here, the 20 Angstrom one, whose reciprocal
    length is smallest) exhausts its enumerated quota, adjacent candidate
    distances can span several meshes and probing the midpoint only catches
    one of them. build_kmesh_entries must truncate there rather than emit a
    ladder with a hole in it."""
    anisotropic = Structure(
        Lattice.orthorhombic(20.0, 3.0, 3.0),
        ["Si"],
        [[0.0, 0.0, 0.0]],
    )
    candidates = generate_candidate_k_distances(anisotropic, max_kpoints_per_axis=4)

    entries = build_kmesh_entries(anisotropic, candidates)

    assert len(entries) > 1, "truncated to nothing; test no longer exercises a gap"
    for (_, previous), (_, current) in itertools.pairwise(entries):
        assert all(
            current_count - previous_count <= 1
            for previous_count, current_count in zip(previous, current, strict=True)
        )
    # Concrete regression check for this exact anisotropic case: the ladder
    # stops at rung 4 rather than continuing past where axis "a" runs dry.
    assert entries == [
        (0, (1, 1, 1)),
        (1, (1, 2, 2)),
        (2, (1, 3, 3)),
        (3, (1, 4, 4)),
        (4, (1, 5, 5)),
    ]


def test_floating_point_noise_at_an_integer_boundary_does_not_inflate_the_mesh() -> (
    None
):
    """A5: recip_length / k_distance can land a few ULPs above an integer due
    to floating-point representation (here, 7.000000000000001 for a cubic
    3.3 Angstrom cell probed at exactly its 1/7 boundary). Without rounding,
    ceil() would push the count to 8; k_distance_to_mesh's ceil(round(x, 5))
    must snap it back to 7."""
    lattice_length = 3.3
    structure = Structure(Lattice.cubic(lattice_length), ["Si"], [[0.0, 0.0, 0.0]])
    recip_length = 2.0 * math.pi / lattice_length
    boundary_k_distance = recip_length / 7
    raw_quotient = recip_length / boundary_k_distance
    assert math.ceil(raw_quotient) == 8, (
        "fixture assumption broken: this k_distance no longer lands just "
        "above the integer boundary it's meant to exercise"
    )

    mesh = k_distance_to_mesh(structure, boundary_k_distance)

    assert mesh == (7, 7, 7)
