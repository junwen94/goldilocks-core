"""Utilities for converting between k-point representations."""

from __future__ import annotations

import math

from pymatgen.core import Structure


def k_distance_to_mesh(
    structure: Structure,
    k_distance: float,
    *,
    force_parity: bool = False,
) -> tuple[int, int, int]:
    """Convert a reciprocal-space k-point distance into a uniform mesh.

    The distance is interpreted as the maximum spacing between adjacent
    k-points along a reciprocal lattice direction, in units of 1/Angstrom.

    Notes
    -----
    The current implementation computes the base mesh from reciprocal lattice
    lengths and does not yet apply the ``force_parity`` option.
    """
    reciprocal_lattice = structure.lattice.reciprocal_lattice
    reciprocal_lengths = (
        reciprocal_lattice.a,
        reciprocal_lattice.b,
        reciprocal_lattice.c,
    )

    mesh = tuple(
        max(1, math.ceil(round(length / k_distance, 5)))
        for length in reciprocal_lengths
    )

    return mesh


def generate_candidate_k_distances(
    structure: Structure,
    max_index: int = 30,
) -> list[float]:
    """Generate candidate k-distance values from reciprocal lattice lengths."""
    reciprocal_lattice = structure.lattice.reciprocal_lattice
    reciprocal_lengths = (
        reciprocal_lattice.a,
        reciprocal_lattice.b,
        reciprocal_lattice.c,
    )

    candidates = {
        round(length / index, 8)
        for length in reciprocal_lengths
        for index in range(1, max_index + 1)
    }

    return sorted(candidates, reverse=True)


def build_k_distance_intervals(
    structure: Structure,
    candidate_distances: list[float],
) -> list[tuple[tuple[int, int, int], float, float]]:
    """Build finite k-distance intervals and their corresponding meshes.

    Notes
    -----
    The returned intervals include the top unbounded interval and the finite
    intervals between adjacent candidate k-distances. The lower tail below the
    smallest candidate distance is intentionally not included.
    """
    intervals: list[tuple[tuple[int, int, int], float, float]] = []

    max_candidate = candidate_distances[0]
    top_probe = max_candidate + 1.0
    top_mesh = k_distance_to_mesh(structure, top_probe)
    intervals.append((top_mesh, max_candidate, math.inf))

    for upper, lower in zip(candidate_distances[:-1], candidate_distances[1:]):
        probe = 0.5 * (upper + lower)
        mesh = k_distance_to_mesh(structure, probe)
        intervals.append((mesh, lower, upper))

    return intervals
