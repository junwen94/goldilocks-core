"""geometry: dimensionality (3d/2d/1d/molecule) from a bond graph, the
lattice vectors along which a cell has vacuum, and the frame of a slab.

The dimensionality is Larsen's connectivity rank
(``pymatgen.analysis.dimensionality.get_dimensionality_larsen``) of
``bond_graph``, which joins two bonding rules:

- ``JmolNN`` bonds a pair closer than the sum of two fixed, covalent-scale
  radii plus 0.45 A. It replaced v1's ``CrystalNN``, whose fingerprint
  matching raised on exotic coordination (stfc/goldilocks-core#133), but
  its cutoffs miss ionic, metallic and long covalent contacts: alone it
  ranked rock-salt NaCl, bcc Na and phosphorene 'molecule' (#125).
- Each site is also bonded to its nearest-neighbour shell, every neighbour
  closer than 1.1 times that site's shortest contact (the rule of
  pymatgen's ``MinimumDistanceNN(tol=0.1)``), so no atom of a dense solid
  is left unbonded.

Adding edges merges components or closes periodic cycles and never removes
either, so the union can only keep or raise the rank JmolNN alone gives.
Molecular crystals and van der Waals layers keep their rank because each
atom's nearest shell lies inside its own molecule or layer.

The rank cannot tell a molecular crystal from a molecule in a box: CO2 dry
ice ranks 'molecule' but fills its cell (#126). The vacuum fact does not
depend on bonds. Along each lattice vector it is the width of the widest
atom-free slab bounded by planes parallel to the other two vectors, measured
between atom centres. A lattice vector whose width exceeds
``VACUUM_GAP_ANGSTROM`` is a vacuum axis. In the 84 structures of the
2026-10-19 batch, no bulk crystal exceeds 3.60 A (graphite, along c), every
monolayer has at least 15.05 A along c, and the isolated C pair of MC3D
mc3d-6516 has 6.10, 6.10 and 8.59 A. A layer that is not parallel to two
lattice vectors shows no vacuum, so its cell is treated as periodic.

A cell with exactly one vacuum axis is a slab (``SlabGeometry``). Quantum
ESPRESSO's 2D cell constraints work in Cartesian axes: '2Dxy' and '2Dshape'
move only the x and y components of a and b and hold c fixed (QE 7.3
Modules/cell_base.f90:742-749 and :803-808). They keep a slab's shape and
vacuum only when the vacuum lies along c and a, b lie in the xy plane, the
frame ``SlabGeometry.xy_frame`` reports. QE's 2D Coulomb truncation assumes
that frame with c along z as well (PW/src/Coul_cut_2D.f90:72-81). All 15
monolayers of the batch are in that frame, with c along z, while its fcc, bcc
and rhombohedral primitive cells have a and b out of the xy plane.
``ab_in_xy_plane`` and ``c_along_z`` are the one test of each, with one
tolerance, so every caller applies the same.

Disordered structures and classifier failures are ``Unavailable``, like
every other ``analysis/`` fact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from pymatgen.analysis.dimensionality import get_dimensionality_larsen
from pymatgen.analysis.local_env import JmolNN
from pymatgen.core import Lattice, Structure
from pymatgen.core.graphs import StructureGraph

from goldilocks_core.resolution import FieldState, Provenance, Resolved, Unavailable

Dimensionality = Literal["3d", "2d", "1d", "molecule"]
LatticeAxis = Literal["a", "b", "c"]

_LATTICE_AXES: tuple[LatticeAxis, LatticeAxis, LatticeAxis] = ("a", "b", "c")

_BY_VALUE: dict[int, Dimensionality] = {3: "3d", 2: "2d", 1: "1d", 0: "molecule"}

_SHELL_TOLERANCE = 0.1
"""A site is bonded to every neighbour closer than (1 + 0.1) times its
shortest contact (``MinimumDistanceNN``'s default ``tol``)."""

_SHELL_SEARCH_RADIUS_ANGSTROM = 6.0
"""Radius searched for each site's nearest-neighbour shell. It holds the
whole shell of the sparsest elemental solid (bcc Cs: shortest contact
5.32 A, shell edge 5.85 A). A site with no neighbour inside it stays
unbonded; ``MinimumDistanceNN``'s 10 A default would bond a lone atom in a
7-9.5 A box to its own periodic images."""

VACUUM_GAP_ANGSTROM = 5.0
"""Atom-free slab width above which a lattice vector is a vacuum axis:
between the widest gap in a bulk crystal of the batch (3.60 A) and the
narrowest vacuum of the isolated C pair (6.10 A)."""

_XY_PLANE_TOLERANCE = 1e-8
"""Largest z component, as a fraction of |a|, of a lattice vector in the xy
plane. QE 7.3 applies the same test before its 2D Coulomb cutoff, to lattice
vectors in units of alat (PW/src/Coul_cut_2D.f90:72-74); alat is |a| for
CELL_PARAMETERS in angstrom (Modules/cell_base.f90:222)."""


@dataclass(frozen=True, slots=True)
class SlabGeometry:
    """A cell with exactly one vacuum axis. ``xy_frame``: that axis is c and
    a, b lie in the Cartesian xy plane, the frame QE's 2D cell constraints
    assume."""

    vacuum_axis: LatticeAxis
    xy_frame: bool


@dataclass(frozen=True, slots=True)
class GeometryFacts:
    """``vacuum_widths_angstrom`` holds the widest atom-free slab along a, b
    and c, in A rounded to 0.01; ``vacuum_axes`` names those wider than
    ``VACUUM_GAP_ANGSTROM``, in a, b, c order. ``slab`` is set only when
    there is exactly one vacuum axis."""

    dimensionality: Dimensionality
    low_dimensional: bool
    vacuum_axes: tuple[LatticeAxis, ...]
    vacuum_widths_angstrom: tuple[float, float, float]
    slab: SlabGeometry | None


def bond_graph(structure: Structure) -> StructureGraph:
    """Bond graph of an ordered ``structure``: ``JmolNN``'s bonds plus, for
    every site, a bond to each neighbour closer than
    ``1 + _SHELL_TOLERANCE`` times the site's shortest contact within
    ``_SHELL_SEARCH_RADIUS_ANGSTROM``. A site with no neighbour within that
    radius adds no bonds."""
    graph = StructureGraph.from_local_env_strategy(structure, JmolNN())
    neighbours_by_site = structure.get_all_neighbors(_SHELL_SEARCH_RADIUS_ANGSTROM)
    for index, neighbours in enumerate(neighbours_by_site):
        if not neighbours:
            continue
        shortest = min(neighbour.nn_distance for neighbour in neighbours)
        shell_edge = (1 + _SHELL_TOLERANCE) * shortest
        for neighbour in neighbours:
            if neighbour.nn_distance < shell_edge:
                graph.add_edge(
                    index,
                    neighbour.index,
                    to_jimage=neighbour.image,
                    warn_duplicates=False,
                )
    return graph


def geometry(structure: Structure) -> FieldState[GeometryFacts]:
    if not structure.is_ordered:
        return Unavailable(
            reason="dimensionality classification does not support "
            "disordered structures"
        )
    try:
        dim_value = get_dimensionality_larsen(bond_graph(structure))
    except (ValueError, RuntimeError) as error:
        return Unavailable(reason=f"dimensionality classification failed: {error}")
    dimensionality = _BY_VALUE.get(dim_value)
    if dimensionality is None:
        return Unavailable(
            reason=f"dimensionality classification returned an unrecognised "
            f"connectivity rank {dim_value!r}"
        )
    widths = vacuum_widths(structure)
    axes = _axes_wider_than_the_gap(widths)
    facts = GeometryFacts(
        dimensionality=dimensionality,
        low_dimensional=bool(dim_value < 3),
        vacuum_axes=axes,
        vacuum_widths_angstrom=widths,
        slab=_slab(structure.lattice, axes),
    )
    return Resolved(facts, Provenance(source="heuristic"))


def vacuum_widths(structure: Structure) -> tuple[float, float, float]:
    """Widest atom-free slab along each lattice vector, in A rounded to 0.01:
    the largest gap between consecutive fractional coordinates along that
    vector, wrapped into the cell, times the spacing 1/|b_i| of the planes
    parallel to the other two vectors (b_i is the reciprocal vector)."""
    reciprocal_lengths = structure.lattice.reciprocal_lattice_crystallographic.abc
    widths = []
    for axis, reciprocal_length in enumerate(reciprocal_lengths):
        coordinates = np.sort(np.mod(structure.frac_coords[:, axis], 1.0))
        gaps = np.diff(coordinates, append=coordinates[0] + 1.0)
        widths.append(round(float(gaps.max()) / reciprocal_length, 2))
    return widths[0], widths[1], widths[2]


def vacuum_axes(structure: Structure) -> tuple[LatticeAxis, ...]:
    """The lattice vectors of ``structure`` whose widest atom-free slab is
    wider than ``VACUUM_GAP_ANGSTROM``, in a, b, c order."""
    return _axes_wider_than_the_gap(vacuum_widths(structure))


def slab_geometry(structure: Structure) -> SlabGeometry | None:
    """``structure``'s vacuum axis and frame, or ``None`` unless it has
    exactly one vacuum axis."""
    return _slab(structure.lattice, vacuum_axes(structure))


def ab_in_xy_plane(lattice: Lattice) -> bool:
    """Whether a and b lie in the Cartesian xy plane: their z components are
    at most ``_XY_PLANE_TOLERANCE`` times |a|."""
    a, b, _ = lattice.matrix
    limit = _XY_PLANE_TOLERANCE * float(np.linalg.norm(a))
    return bool(abs(a[2]) <= limit and abs(b[2]) <= limit)


def c_along_z(lattice: Lattice) -> bool:
    """Whether c lies along the Cartesian z axis: its x and y components are
    at most ``_XY_PLANE_TOLERANCE`` times |a|, the tolerance of
    ``ab_in_xy_plane``."""
    a, _, c = lattice.matrix
    limit = _XY_PLANE_TOLERANCE * float(np.linalg.norm(a))
    return bool(abs(c[0]) <= limit and abs(c[1]) <= limit)


def _slab(lattice: Lattice, axes: tuple[LatticeAxis, ...]) -> SlabGeometry | None:
    if len(axes) != 1:
        return None
    (axis,) = axes
    return SlabGeometry(
        vacuum_axis=axis, xy_frame=axis == "c" and ab_in_xy_plane(lattice)
    )


def _axes_wider_than_the_gap(
    widths: tuple[float, float, float],
) -> tuple[LatticeAxis, ...]:
    return tuple(
        axis
        for axis, width in zip(_LATTICE_AXES, widths, strict=True)
        if width > VACUUM_GAP_ANGSTROM
    )
