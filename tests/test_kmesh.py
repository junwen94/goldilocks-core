import math

from pymatgen.core import Lattice, Structure

from goldilocks_core.processing.kmesh import (
    build_k_distance_intervals,
    generate_candidate_k_distances,
    k_distance_to_mesh,
)


def test_k_distance_to_mesh_returns_expected_uniform_mesh() -> None:
    """Convert a reciprocal-space distance into a uniform mesh."""
    structure = Structure(
        lattice=Lattice.cubic(3.5),
        species=["Si"],
        coords=[[0.0, 0.0, 0.0]],
    )

    mesh = k_distance_to_mesh(structure, k_distance=1.0)

    assert mesh == (2, 2, 2)


def test_generate_candidate_k_distances_returns_sorted_values() -> None:
    """Generate reciprocal-length-based candidate k-distances."""
    structure = Structure(
        lattice=Lattice.cubic(3.5),
        species=["Si"],
        coords=[[0.0, 0.0, 0.0]],
    )

    candidates = generate_candidate_k_distances(structure, max_index=3)

    reciprocal_length = structure.lattice.reciprocal_lattice.a

    assert len(candidates) > 0
    assert candidates == sorted(candidates, reverse=True)
    assert math.isclose(candidates[0], round(reciprocal_length / 1, 8))
    assert math.isclose(candidates[-1], round(reciprocal_length / 3, 8))


def test_build_k_distance_intervals_records_mesh_intervals() -> None:
    """Build k-distance intervals and their corresponding meshes."""
    structure = Structure(
        lattice=Lattice.cubic(3.5),
        species=["Si"],
        coords=[[0.0, 0.0, 0.0]],
    )

    candidates = generate_candidate_k_distances(structure, max_index=4)
    intervals = build_k_distance_intervals(structure, candidates)

    assert len(intervals) > 0
    assert intervals[0][0] == (1, 1, 1)
    assert math.isinf(intervals[0][2])
    assert any(mesh == (2, 2, 2) for mesh, _, _ in intervals)
    assert any(mesh == (3, 3, 3) for mesh, _, _ in intervals)
    assert any(mesh == (4, 4, 4) for mesh, _, _ in intervals)
