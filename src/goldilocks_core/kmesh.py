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

**Rung 1's open bound is ``None``.** goldilocks-data stores the upper
k-distance bound of rung 1, the Gamma-only mesh, as ``math.inf``. Core
stores ``None``: strict JSON has no infinity, and the ``inf`` in a
Gamma-only k_sampling record made ``/run`` fail and wrote ``Infinity``
into ``goldilocks.json`` (#137).

**A vacuum axis gets one division.** goldilocks-data divides all three
reciprocal lengths by the k-distance, which gave a monolayer two or three
k-points across its vacuum and a molecule in a box a 3x3x2 mesh, for no
change in energy and up to twice the cost for a monolayer, six times for
the box (#159). Here every lattice vector of the structure passed in that
``analysis.geometry.vacuum_axes`` names (an atom-free slab wider than
5 A) has one division in every mesh, whatever the k-distance. Only
periodic axes give change points, so a slab's rungs step its periodic
axes alone and are numbered differently from goldilocks-data's, and a
cell with vacuum along all three axes has one rung, Gamma-only at every
k-distance. A cell without vacuum keeps goldilocks-data's ladder rung for
rung. The vacuum is read from the cell the mesh is built for, never from
the dimensionality label, which calls bulk graphite '2d'; ``_Axes`` reads
it once per call, so a ladder reads it once.

**Naming: no ``k_spacing``.** v1 conflated two different reciprocal-length
conventions -- the solid-state (2*pi) convention (here, ``k_distance``,
matching AiiDA-QuantumESPRESSO) and the crystallographic (no 2*pi)
convention (here, ``k_line_density``) -- into one ambiguous ``k_spacing``
name. This module uses only ``k_distance``/``k_line_density``; the name
``k_spacing`` does not appear anywhere in v2.

**Not ported**: ``n_reduced_kpoints`` (above); goldilocks-data's own
``kindex_points`` helper, which builds
``goldilocks_data.sweeps.models.SweepPoint`` objects for its own
sweep-generation glue -- core has no such concept and must not take a
dependency on the ``goldilocks_data`` package; and its public
``generate_candidate_k_distances`` and ``mesh_to_k_line_density_interval``,
which core computes only inside ``build_gamma_kmesh_entries`` (``_Axes``).
Everything else here is the actual k-mesh math, kept behaviourally
identical apart from rung 1's ``None`` bound and the vacuum axes (above);
only the loose ``Any`` structure type hint is tightened to pymatgen's own
``Structure`` (core, unlike data, already requires pymatgen
unconditionally), and one ``zip(xs[:-1], xs[1:])`` pairing is rewritten as
``itertools.pairwise`` to satisfy core's lint config -- both mechanical,
behaviour-preserving.

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

from pymatgen.core import Lattice, Structure

from goldilocks_core.analysis.geometry import LATTICE_AXES, vacuum_axes

MIN_K_DISTANCE = 0.03
"""Resolution floor for the k-mesh ladder, in Angstrom^-1 on the
solid-state (2*pi) reciprocal lattice -- the same convention as
AiiDA-QuantumESPRESSO k-distance. Change points denser than this are not
enumerated, and this value is part of every recorded ``kindex``: a rung
means nothing without it."""


@dataclass(frozen=True, slots=True)
class KMeshEntry:
    """One gamma-centered k-mesh entry in increasing kindex order.

    ``k_distance_interval`` is the ``(low, high)`` range of k-distances,
    in Angstrom^-1, that give ``mesh``. ``high`` is ``None`` only for rung
    1, the Gamma-only mesh, which every k-distance above ``low`` gives;
    ``low`` is 0 for a structure with no periodic axis.
    ``k_line_density_interval`` is the range of scalar k-line densities, in
    Angstrom^-1 on the crystallographic reciprocal lattice, that give
    ``mesh`` along every periodic axis, or ``None`` when no single value
    does or there is no periodic axis."""

    kindex: int
    mesh: tuple[int, int, int]
    k_distance_interval: tuple[float, float | None]
    k_line_density_interval: tuple[float, float] | None
    k_pra: float


@dataclass(frozen=True, slots=True)
class _Axes:
    """The lattice vectors of a structure as the mesh sees them: the
    reciprocal lengths, in Angstrom^-1 on the solid-state (2*pi) reciprocal
    lattice (``lengths``) and on the crystallographic one (``line_lengths``),
    and whether each vector is periodic, that is, not one of the structure's
    ``analysis.geometry.vacuum_axes``. Built once per call of
    ``k_distance_to_mesh`` or ``build_gamma_kmesh_entries``, so a whole ladder
    reads the vacuum once."""

    lengths: tuple[float, float, float]
    line_lengths: tuple[float, float, float]
    periodic: tuple[bool, bool, bool]

    @classmethod
    def from_structure(cls, structure: Structure) -> _Axes:
        vacuum = vacuum_axes(structure)
        a, b, c = (axis not in vacuum for axis in LATTICE_AXES)
        return cls(
            lengths=_abc(structure.lattice.reciprocal_lattice),
            line_lengths=_abc(structure.lattice.reciprocal_lattice_crystallographic),
            periodic=(a, b, c),
        )

    def mesh(self, k_distance: float) -> tuple[int, int, int]:
        """``ceil(round(|b_i| / k_distance, 5))`` divisions, at least one,
        along each periodic axis, and one along each vacuum axis."""
        a, b, c = (
            max(1, math.ceil(round(length / k_distance, 5))) if periodic else 1
            for length, periodic in zip(self.lengths, self.periodic, strict=True)
        )
        return (a, b, c)

    def change_points(self, min_k_distance: float) -> list[float]:
        """Every k-distance at which some periodic axis changes its k-point
        count, largest first: ``mesh_i = ceil(|b_i| / k_distance)`` steps
        from ``n`` to ``n + 1`` exactly at ``|b_i| / n``, rounded here to 8
        decimals. A vacuum axis has one division at every k-distance and
        gives none (#159). ``min_k_distance``, the same floor for every axis,
        bounds the quotients, so all axes run out of change points together
        and none above the floor is skipped."""
        return sorted(
            {
                round(length / index, 8)
                for length, periodic in zip(self.lengths, self.periodic, strict=True)
                if periodic
                for index in range(1, max(1, math.floor(length / min_k_distance)) + 1)
            },
            reverse=True,
        )

    def line_density_interval(
        self, mesh: tuple[int, int, int]
    ) -> tuple[float, float] | None:
        """The intersection of the periodic axes' intervals
        ``[(n_i - 0.5) / |b_i|, (n_i + 0.5) / |b_i|]``, or ``None`` when it is
        empty or there is no periodic axis."""
        sampled = [
            (nk, length)
            for nk, length, periodic in zip(
                mesh, self.line_lengths, self.periodic, strict=True
            )
            if periodic
        ]
        if not sampled:
            return None
        lower = max(max(0.0, (nk - 0.5) / length) for nk, length in sampled)
        upper = min((nk + 0.5) / length for nk, length in sampled)
        if lower > upper:
            return None
        return (float(lower), float(upper))


def _abc(reciprocal: Lattice) -> tuple[float, float, float]:
    return (float(reciprocal.a), float(reciprocal.b), float(reciprocal.c))


def k_distance_to_mesh(structure: Structure, k_distance: float) -> tuple[int, int, int]:
    """Convert a solid-state reciprocal spacing in Angstrom^-1 to a mesh:
    ``ceil(round(|b_i| / k_distance, 5))`` divisions, at least one, along
    each periodic lattice vector of ``structure``, and exactly one along
    each of its vacuum axes, whatever ``k_distance`` is (#159)."""
    return _Axes.from_structure(structure).mesh(k_distance)


def build_gamma_kmesh_entries(
    structure: Structure, min_k_distance: float = MIN_K_DISTANCE
) -> list[KMeshEntry]:
    """Build the unshifted, Gamma-inclusive k-mesh ladder for a structure.

    ``kindex`` is 1-based and rung 1 is the Gamma-only ``(1, 1, 1)`` mesh,
    which the first probe always yields because it sits above every
    periodic ``|b_i|``. Rung 1's interval is open above, so its upper bound
    is ``None``: a float ``inf`` there cannot be written as strict JSON
    (#137).

    Every rung has one division along each vacuum axis, and only the
    periodic axes step (#159). A structure with vacuum along all three axes
    has no change point, so rung 1, every k-distance above 0, is its only
    rung.

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
    axes = _Axes.from_structure(structure)
    candidates = axes.change_points(min_k_distance)
    # Rung 1 holds above the largest change point, or at every k-distance
    # when no periodic axis gives one.
    top = candidates[0] if candidates else 0.0
    intervals: list[tuple[tuple[int, int, int], tuple[float, float | None]]] = [
        (axes.mesh(top + 1.0), (top, None))
    ]
    for upper, lower in pairwise(candidates):
        intervals.append((axes.mesh(0.5 * (upper + lower)), (lower, upper)))

    entries: list[KMeshEntry] = []
    seen: set[tuple[int, int, int]] = set()
    for mesh, interval in intervals:
        if mesh in seen:
            continue
        seen.add(mesh)
        entries.append(
            KMeshEntry(
                kindex=len(entries) + 1,
                mesh=mesh,
                k_distance_interval=interval,
                k_line_density_interval=axes.line_density_interval(mesh),
                k_pra=float(len(structure) * mesh[0] * mesh[1] * mesh[2]),
            )
        )
    return entries
