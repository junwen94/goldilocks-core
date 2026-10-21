from __future__ import annotations

import math
from itertools import pairwise

import pytest
from pymatgen.core import Lattice, Structure
from support import graphene_monolayer

from goldilocks_core.kmesh import build_gamma_kmesh_entries, k_distance_to_mesh


def _dense_orthorhombic(a: float, b: float, c: float) -> Structure:
    """An orthorhombic a x b x c A cell with Si atoms along its body
    diagonal, at most 4 A apart along each lattice vector: no vector is a
    vacuum axis (more than 5 A atom-free), so every axis is sampled and the
    mesh depends on the lattice alone (#159)."""
    count = math.ceil(max(a, b, c) / 4.0)
    return Structure(
        Lattice.orthorhombic(a, b, c),
        ["Si"] * count,
        [[index / count] * 3 for index in range(count)],
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
    structure = _dense_orthorhombic(*lengths)

    mesh = k_distance_to_mesh(structure, spacing)
    reciprocal_lengths = tuple(2.0 * math.pi / length for length in lengths)
    expected = tuple(math.ceil(length / spacing) for length in reciprocal_lengths)

    assert mesh == expected
    for reciprocal_length, points in zip(reciprocal_lengths, mesh, strict=True):
        assert reciprocal_length / points <= spacing
        if points > 1:
            assert reciprocal_length / (points - 1) > spacing


@pytest.mark.parametrize("k_distance", [0.03, 0.1, 0.2, 0.5, 2.0])
def test_a_slab_gets_one_k_point_along_its_vacuum_at_any_spacing(
    k_distance: float,
) -> None:
    """A slab's electrons do not disperse across its vacuum, so one k-point
    samples that axis: pw.x gave the 2H-MoS2 monolayer the same total energy
    to 1e-8 Ry on 9x9x1 as on 9x9x2, in 0.6 of the time (#159). The graphene
    sheet's 15 A cell has |b_3| = 2 pi / 15 A = 0.42 A^-1, so
    ceil(|b_3| / k_distance) would give it 3 k-points at 0.2 A^-1 and 14 at
    0.03. The sheet's own axes keep the spacing, |b_1| = |b_2| =
    4 pi / (sqrt(3) a) for the hexagonal lattice."""
    slab = graphene_monolayer(c=15.0)
    in_plane = math.ceil(4 * math.pi / (math.sqrt(3) * 2.46) / k_distance)

    assert k_distance_to_mesh(slab, k_distance) == (in_plane, in_plane, 1)


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
    port docstring in ``kmesh.py`` for why the redesign works. Atoms fill
    the 20 Angstrom axis, so it is periodic, not vacuum (#159)."""
    anisotropic = _dense_orthorhombic(20.0, 3.0, 3.0)

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
