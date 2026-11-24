"""enumerate_orderings: the distinct collinear magnetic orderings of a
structure's magnetic sites, from spinforge's spin-symmetry-adapted (SSA)
structures (T. Nomoto, K. Shinohara, H. Watanabe and R. Arita, Phys. Rev. X
16, 041052 (2026); #308).

spinforge enumerates, for each spin space group of the crystal whose
invariant translation lattice has index 1 or 2 (``_K_INDICES``, the
primitive cell or twice it), the moments that group allows, as a basis per
group of symmetry-equivalent sites. Its defaults are used unmodified
(Junwen, 2026-11-24): only subgroups that keep the magnetic sites'
multiplicity, searched to its default depth, one per class of conjugate
subgroups.

Each SSA structure becomes collinear orderings like this:
- every site group carries one basis vector, along one spin axis, which
  fixes the relative signs inside the group;
- the relative sign between groups is free, so each sign combination up to
  a global flip is one ordering;
- a structure whose groups do not cover every magnetic site forces some
  moments to zero and is dropped.

All-parallel orderings are ferromagnetic and dropped (the caller lists its
own ``fm``). The rest are reduced to their primitive magnetic cell and
deduplicated: two orderings are the same when the crystal's symmetry or a
global spin flip maps one onto the other. A ``k_index`` 2 enumeration
repeats the ``k_index`` 1 orderings in a doubled cell, so the reduction is
what keeps each once.

An ordering is ``"afm"`` when some operation of its cell combined with time
reversal maps every moment onto the reversed moment of a site of the same
element, so the moments cancel by symmetry, and ``"fim"`` (ferrimagnetic)
otherwise. Equal spin counts are not enough: A up and B down with two
different magnetic elements is ferrimagnetic.

The enumeration runs on one long-lived worker process
(``worker_process.WorkerProcess``), one at a time, so spinforge and its
group-theory stack are imported there and not in the caller.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Literal, NamedTuple

from goldilocks_core.worker_process import WorkerProcess

if TYPE_CHECKING:
    import numpy as np
    from pymatgen.core import PeriodicSite, Structure
    from spinforge.configuration import SpinSymmetryAdaptedStructure

OrderingKind = Literal["afm", "fim"]

_K_INDICES = (1, 2)
"""spinforge's ``k_index`` values searched: the invariant translation lattice
is the primitive lattice or a sublattice of index 2 (Junwen, 2026-11-24)."""

_SIGN_TOLERANCE = 1e-6
"""Below this, a projected moment counts as zero."""

_ENUMERATION_WORKER = WorkerProcess()


class EnumeratedOrdering(NamedTuple):
    """One distinct ordering: its primitive magnetic cell, each magnetic site
    a ``Species`` with ``spin`` +1 or -1 and every other site a ``Species``
    without spin, and whether its moments cancel by symmetry."""

    structure: Structure
    kind: OrderingKind


def enumerate_orderings(
    structure: Structure, sublattice: tuple[str, ...], symprec: float
) -> tuple[EnumeratedOrdering, ...]:
    """Every distinct antiferromagnetic and ferrimagnetic collinear ordering
    of the ``sublattice`` elements' sites in ``structure`` (module
    docstring), in no particular order; empty when there is none.
    ``symprec`` is the length tolerance, in Angstrom, for the crystal's
    symmetry and its primitive cell.

    Raises what spinforge raises, re-raised from the worker, and
    ``worker_process.WorkerDied`` if the worker process dies."""
    return _ENUMERATION_WORKER.run(_enumerate, structure, sublattice, symprec)


def _enumerate(
    structure: Structure, sublattice: tuple[str, ...], symprec: float
) -> tuple[EnumeratedOrdering, ...]:
    """Worker side of ``enumerate_orderings``."""
    from loguru import logger
    from moyopy import MoyoDataset
    from moyopy.interface import MoyoAdapter
    from pymatgen.core import Element
    from spinforge.configuration import SSAGenerator
    from spinspg.spin import SpinOnlyGroupType

    logger.disable("spinforge")
    dataset = MoyoDataset(
        MoyoAdapter.from_structure(structure), symprec=symprec, rotate_basis=False
    )
    primitive = dataset.prim_std_cell
    magnetic = [
        index
        for index, number in enumerate(primitive.numbers)
        if Element.from_Z(number).symbol in sublattice
    ]
    generator = SSAGenerator(
        prim_cell=primitive, magnetic_site_indices=magnetic, symprec=symprec
    )
    orderings: list[Structure] = []
    for k_index in _K_INDICES:
        for _spin_only, _spin_space_group, adapted in generator.enumerate(
            spin_only_group_type=SpinOnlyGroupType.COLLINEAR, k_index=k_index
        ):
            orderings.extend(_orderings_of(adapted, sublattice))
    distinct = _distinct(orderings, sublattice, symprec)
    return tuple(
        EnumeratedOrdering(
            ordering, "afm" if _compensated(ordering, symprec) else "fim"
        )
        for ordering in distinct
    )


def _orderings_of(
    adapted: SpinSymmetryAdaptedStructure, sublattice: tuple[str, ...]
) -> list[Structure]:
    """The collinear orderings one SSA structure allows: one per sign of each
    site group relative to the first, none when its groups leave a magnetic
    site without a moment."""
    import numpy as np

    groups = list(adapted.grouped_magnetic_moments_basis.items())
    cell = adapted.generate_with_magnetic_moments(
        np.zeros((len(adapted.supercell.numbers), 3))
    )
    magnetic = {
        index for index, site in enumerate(cell) if site.specie.symbol in sublattice
    }
    covered = set().union(*(sites for sites, _indices in groups))
    if not groups or covered != magnetic:
        return []
    vectors = [adapted.magnetic_moments_basis[indices[0]] for _sites, indices in groups]
    axis = next(row for row in vectors[0] if np.linalg.norm(row) > _SIGN_TOLERANCE)
    axis = axis / np.linalg.norm(axis)
    orderings = []
    for flips in itertools.product((1.0, -1.0), repeat=len(vectors) - 1):
        projected = sum(
            sign * (vector @ axis)
            for sign, vector in zip((1.0, *flips), vectors, strict=True)
        )
        orderings.append(_decorated(cell, _signs(projected)))
    return orderings


def _signs(projected: np.ndarray) -> tuple[float, ...]:
    return tuple(
        0.0 if abs(value) < _SIGN_TOLERANCE else float(1 if value > 0 else -1)
        for value in projected
    )


def _decorated(cell: Structure, signs: tuple[float, ...]) -> Structure:
    """``cell`` with each site a ``Species`` carrying its spin sign, none on
    a site whose sign is 0."""
    from pymatgen.core import Species, Structure

    species = [
        Species(site.specie.symbol, spin=sign) if sign else Species(site.specie.symbol)
        for site, sign in zip(cell, signs, strict=True)
    ]
    return Structure(cell.lattice, species, cell.frac_coords)


_STAND_IN_ELEMENTS = ("He", "Ne", "Ar", "Kr", "Xe", "Rn", "Es", "Fm", "Md", "No")
"""Elements that stand for a magnetic element's spin-down sites when two
orderings are compared (``_coloured``); the first ones absent from the
structure are used."""


def _distinct(
    orderings: list[Structure], sublattice: tuple[str, ...], symprec: float
) -> list[Structure]:
    """The non-ferromagnetic ``orderings``, each in its primitive magnetic
    cell, one per class of orderings the crystal's symmetry or a global spin
    flip relate. Orderings are compared with each magnetic element's
    spin-down sites as a stand-in element (``_coloured``): pymatgen's
    ``SpinComparator`` only pairs opposite spins, so it never matches an
    ordering in which an element has one spin only (B2 FeCo, Fe up and Co
    down) with itself."""
    from pymatgen.core.structure_matcher import StructureMatcher

    matcher = StructureMatcher(
        primitive_cell=False, attempt_supercell=True, scale=False
    )
    present = {site.specie.symbol for site in orderings[0]} if orderings else set()
    stand_ins = dict(
        zip(
            sorted(sublattice),
            (element for element in _STAND_IN_ELEMENTS if element not in present),
            strict=False,
        )
    )
    kept: dict[int, list[tuple[Structure, Structure]]] = {}
    for ordering in orderings:
        spins = {_spin(site) for site in ordering if site.specie.symbol in sublattice}
        if len(spins) < 2:
            continue
        reduced = ordering.get_primitive_structure(
            tolerance=symprec, use_site_props=False
        )
        coloured = _coloured(reduced, stand_ins, 1.0)
        flipped = _coloured(reduced, stand_ins, -1.0)
        same_size = kept.setdefault(len(reduced), [])
        if not any(
            matcher.fit(coloured, other) or matcher.fit(flipped, other)
            for _ordering, other in same_size
        ):
            same_size.append((reduced, coloured))
    return [ordering for size in sorted(kept) for ordering, _coloured in kept[size]]


def _coloured(ordering: Structure, stand_ins: dict[str, str], up: float) -> Structure:
    """``ordering`` without spins, each magnetic site whose spin times ``up``
    is negative replaced by its element's stand-in, so plain structure
    matching tells orderings apart (``up`` -1 compares the flipped
    ordering)."""
    from pymatgen.core import Structure

    species = [
        stand_ins[site.specie.symbol] if _spin(site) * up < 0 else site.specie.symbol
        for site in ordering
    ]
    return Structure(ordering.lattice, species, ordering.frac_coords)


def _spin(site: PeriodicSite) -> float:
    return float(getattr(site.specie, "spin", None) or 0.0)


def _compensated(ordering: Structure, symprec: float) -> bool:
    """Whether an operation of ``ordering``'s cell combined with time reversal
    maps every site onto a site of the same element with the opposite spin
    (spglib's magnetic symmetry of the spins as collinear moments)."""
    import numpy as np
    import spglib

    dataset = spglib.get_magnetic_symmetry(
        (
            ordering.lattice.matrix,
            ordering.frac_coords,
            [site.specie.Z for site in ordering],
            [_spin(site) for site in ordering],
        ),
        symprec=symprec,
        mag_symprec=_SIGN_TOLERANCE,
        is_axial=False,
    )
    return dataset is not None and bool(np.any(dataset["time_reversals"]))
