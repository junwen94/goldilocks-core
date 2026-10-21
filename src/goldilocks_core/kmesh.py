"""k-point mesh ladder math, ported from goldilocks-data's ``kmesh.py``
(v2 epic 6, #6), replacing v1's ``kmesh/`` package (renamed to
``legacy_kmesh/`` to free this name -- see that package's own docstring).

**Why replace rather than reuse v1's own ladder.** v1's
``generate_candidate_k_distances`` bounded enumeration with a fixed
per-axis count (``max_kpoints_per_axis=50``): axes with different
reciprocal lengths exhaust that quota at *different* distances, leaving
holes in the ladder that v1's ``_complete_meshes`` then silently
papered over by truncating the whole list the first time any axis's
count jumped by more than one between rungs. This module instead bounds
enumeration with a single distance floor (``MIN_K_DISTANCE``, uniform
across every axis), so every axis runs out of change points together and
the ladder is complete and non-repeating down to that floor -- the
truncation logic simply has nothing to trigger it and does not exist
here. v1 also rounded inconsistently between its two entry points
(5 decimals in one, 8 in the other, able to disagree at a boundary);
this module keeps goldilocks-data's own two-stage rounding (8 decimals
to enumerate candidate change points, 5 decimals when converting any
given probe distance to an integer mesh), which stays internally
consistent precisely because the candidate list itself is never
truncated.

**``k_pra``**, which v1's ladder never computed, is added here:
k-points per reciprocal atom, a cross-structure-comparable sampling-density
metric.

**No symmetry reduction.** goldilocks-data's ``KMeshEntry`` also carries
``n_reduced_kpoints``, from one ``SpacegroupAnalyzer.get_ir_reciprocal_mesh``
call per rung. Core's does not: nothing in core reads a rung's reduced
count, and reducing every rung made one ladder build take seconds on a
low-symmetry or small cell (#155). The ladder is arithmetic on the
reciprocal lattice, and ``kindex`` is defined exactly as in
goldilocks-data. ``advisors/n_irr_k.py`` is the one place a mesh is
reduced: the mesh a step actually uses.

**Naming: no ``k_spacing``.** v1 conflated two different reciprocal-length
conventions -- the solid-state (2*pi) convention (here, ``k_distance``,
matching AiiDA-QuantumESPRESSO) and the crystallographic (no 2*pi)
convention (here, ``k_line_density``) -- into one ambiguous ``k_spacing``
name. This module uses only ``k_distance``/``k_line_density``; the name
``k_spacing`` does not appear anywhere in v2.

**Not ported**: ``n_reduced_kpoints`` (above), and goldilocks-data's own
``kindex_points`` helper, which builds
``goldilocks_data.sweeps.models.SweepPoint`` objects for its own
sweep-generation glue -- core has no such concept and must not take a
dependency on the ``goldilocks_data`` package. Everything else here is
the actual k-mesh math, kept behaviourally identical; only the loose
``Any`` structure type hint is tightened to pymatgen's own ``Structure``
(core, unlike data, already requires pymatgen unconditionally), and one
``zip(xs[:-1], xs[1:])`` pairing is rewritten as ``itertools.pairwise``
to satisfy core's lint config -- both mechanical, behaviour-preserving.

The ``is_metal``-based k_distance ceiling (metals >= 0.06 A^-1,
semiconductors/insulators >= 0.12 A^-1 as a cost safety net) and the
rung-can't-exceed-this-structure's-own-ladder-length clamp
(goldilocks-core-design.md:1732-1817) both gate *inputs into* this
ladder from the future ML tier (v2 epic 11) and are out of scope here --
this module only builds the ladder itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

from pymatgen.core import Structure

MIN_K_DISTANCE = 0.03
"""Resolution floor for the k-mesh ladder, in Angstrom^-1 on the
solid-state (2*pi) reciprocal lattice -- the same convention as
AiiDA-QuantumESPRESSO k-distance. Change points denser than this are not
enumerated, and this value is part of every recorded ``kindex``: a rung
means nothing without it."""


@dataclass(frozen=True, slots=True)
class KMeshEntry:
    """One gamma-centered k-mesh entry in increasing kindex order."""

    kindex: int
    mesh: tuple[int, int, int]
    k_distance_interval: tuple[float, float]
    k_line_density_interval: tuple[float, float] | None
    k_pra: float


def _reciprocal_lengths(
    structure: Structure, *, crystallographic: bool = False
) -> tuple[float, float, float]:
    lattice = structure.lattice
    reciprocal = (
        lattice.reciprocal_lattice_crystallographic
        if crystallographic
        else lattice.reciprocal_lattice
    )
    return (float(reciprocal.a), float(reciprocal.b), float(reciprocal.c))


def k_distance_to_mesh(structure: Structure, k_distance: float) -> tuple[int, int, int]:
    """Convert a solid-state reciprocal spacing in Angstrom^-1 to a mesh."""
    lengths = _reciprocal_lengths(structure)
    return tuple(max(1, math.ceil(round(length / k_distance, 5))) for length in lengths)


def generate_candidate_k_distances(
    structure: Structure, min_k_distance: float = MIN_K_DISTANCE
) -> list[float]:
    """Return every k-distance at which some axis changes its k-point count.

    ``mesh_i = ceil(|b_i| / k_distance)`` steps from ``n`` to ``n + 1``
    exactly at ``k_distance = |b_i| / n``, so those quotients are the only
    distances where the mesh can change.

    ``min_k_distance`` is the resolution floor in Angstrom^-1 on the
    solid-state (2*pi) reciprocal lattice; quotients below it are not
    enumerated. The floor is identical for every axis, so all axes run out
    of change points together and the returned list is a complete set of
    change points over the whole ``[min_k_distance, inf)`` range -- there
    is no region where a reachable mesh is silently skipped.
    """
    lengths = _reciprocal_lengths(structure)
    return sorted(
        {
            round(length / index, 8)
            for length in lengths
            for index in range(1, max(1, math.floor(length / min_k_distance)) + 1)
        },
        reverse=True,
    )


def mesh_to_k_line_density_interval(
    structure: Structure, mesh: tuple[int, int, int]
) -> tuple[float, float]:
    """Return the scalar k-line-density interval that maps to ``mesh``."""
    lengths = _reciprocal_lengths(structure, crystallographic=True)
    lower = max(
        max(0.0, (nk - 0.5) / length) for nk, length in zip(mesh, lengths, strict=True)
    )
    upper = min((nk + 0.5) / length for nk, length in zip(mesh, lengths, strict=True))
    if lower > upper:
        raise ValueError(f"No scalar k-line-density interval for mesh={mesh}")
    return (float(lower), float(upper))


def build_gamma_kmesh_entries(
    structure: Structure, min_k_distance: float = MIN_K_DISTANCE
) -> list[KMeshEntry]:
    """Build the unshifted, Gamma-inclusive k-mesh ladder for a structure.

    ``kindex`` is 1-based and rung 1 is the Gamma-only ``(1, 1, 1)`` mesh,
    which the first probe always yields because it sits above every
    ``|b_i|``.

    The rung is an ordinal position on this structure's ladder and counts
    nothing. It equals the densest axis count only where the axes step
    together, as in a cubic cell.

    The ladder is complete and non-repeating down to ``min_k_distance``:
    every change point above the floor is enumerated, so consecutive
    rungs differ by at most one k-point on each axis and no reachable
    mesh is skipped.

    A mesh already on the ladder is dropped. Two axes of *almost* equal
    length put two change points a hair apart, and the sliver between
    them can round to the same mesh as its neighbour; without the skip
    that mesh would take two ``kindex`` values. Exactly equal axes do not
    do this -- their quotients are identical and collapse in the
    candidate set -- so it takes a near miss, and it is rare: 36 of the
    20,826 MC3D structures in goldilocks-data's SCF campaign hit it.

    Arithmetic only: no rung is reduced by symmetry
    (``advisors/n_irr_k.py`` reduces the mesh a step uses).
    """
    candidates = generate_candidate_k_distances(structure, min_k_distance)
    if not candidates:
        return []

    intervals = [
        (
            k_distance_to_mesh(structure, candidates[0] + 1.0),
            (candidates[0], math.inf),
        )
    ]
    for upper, lower in pairwise(candidates):
        intervals.append(
            (k_distance_to_mesh(structure, 0.5 * (upper + lower)), (lower, upper))
        )

    entries: list[KMeshEntry] = []
    seen: set[tuple[int, int, int]] = set()
    for mesh, interval in intervals:
        if mesh in seen:
            continue
        seen.add(mesh)
        try:
            line_interval = mesh_to_k_line_density_interval(structure, mesh)
        except ValueError:
            line_interval = None
        entries.append(
            KMeshEntry(
                kindex=len(entries) + 1,
                mesh=mesh,
                k_distance_interval=interval,
                k_line_density_interval=line_interval,
                k_pra=float(len(structure) * mesh[0] * mesh[1] * mesh[2]),
            )
        )
    return entries
