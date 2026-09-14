from __future__ import annotations

import math
from itertools import pairwise

import pytest
from pymatgen.core import Lattice, Structure

from goldilocks_core.kmesh import build_gamma_kmesh_entries, k_distance_to_mesh


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
    """A4, v2 shape (v2 epic 9, #9): v1's ladder bounded enumeration with a
    fixed per-axis count, so axes with different reciprocal lengths ran out
    of change points at different distances -- v1 then truncated the whole
    ladder the first time that produced a hole (a jump of more than one
    k-point on some axis between adjacent rungs), rather than actually
    preventing the hole.

    v2's ``build_gamma_kmesh_entries`` (``kmesh.py``, ported verbatim from
    goldilocks-data in v2 epic 6, #6) removes the need for that truncation
    outright: it bounds enumeration with one distance floor shared by every
    axis (``MIN_K_DISTANCE``), so every axis runs out of change points
    together and no hole is ever produced in the first place -- confirmed
    below over the *entire* ladder for this exact anisotropic case (76
    rungs), not just the handful v1's truncation used to stop at. This is a
    genuine v2 correctness improvement over v1's workaround, not a test
    changed to dodge a v2 regression -- see this repo's own goldilocks-data
    port docstring in ``kmesh.py`` for why the redesign works."""
    anisotropic = Structure(
        Lattice.orthorhombic(20.0, 3.0, 3.0),
        ["Si"],
        [[0.0, 0.0, 0.0]],
    )

    entries = build_gamma_kmesh_entries(anisotropic)

    assert len(entries) > 5, "ladder too short to exercise the axis-a exhaustion point"
    assert entries[0].mesh == (1, 1, 1)
    for previous, current in pairwise(entries):
        assert all(
            current_count - previous_count <= 1
            for previous_count, current_count in zip(
                previous.mesh, current.mesh, strict=True
            )
        )
    # Concrete regression check: axis "a" (the 20 Angstrom one) exhausts its
    # change points first, at rung 8, exactly where v1's truncation used to
    # cut the ladder off -- v2 keeps going instead of stopping there.
    assert entries[6].mesh == (1, 7, 7)
    assert entries[7].mesh == (2, 7, 7)


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
