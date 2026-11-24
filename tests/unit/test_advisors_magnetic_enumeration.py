"""The collinear orderings spinforge's spin-symmetry-adapted structures give
(#308): one per class of symmetry-related patterns in its primitive magnetic
cell, ferromagnetic ones dropped, split by whether the moments cancel by
symmetry. Real enumerations, run through the worker process."""

from __future__ import annotations

import random

import pytest
from pymatgen.core import Lattice, Structure

from goldilocks_core.advisors.magnetic_enumeration import (
    _distinct,
    _raw_orderings,
    _reversed,
    enumerate_orderings,
    site_signature,
)
from goldilocks_core.analysis.symmetry import SYMPREC


def _rock_salt_nio() -> Structure:
    return Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(4.17), ["Ni", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    ).get_primitive_structure()


def _corundum_cr2o3() -> Structure:
    return Structure.from_spacegroup(
        "R-3c",
        Lattice.hexagonal(4.96, 13.6),
        ["Cr", "O"],
        [[0, 0, 0.3475], [0.3056, 0, 0.25]],
    ).get_primitive_structure()


def _rutile_mnf2() -> Structure:
    return Structure.from_spacegroup(
        "P4_2/mnm",
        Lattice.tetragonal(4.87, 3.31),
        ["Mn", "F"],
        [[0, 0, 0], [0.305, 0.305, 0]],
    )


def _spinel_fe3o4() -> Structure:
    return Structure.from_spacegroup(
        "Fd-3m",
        Lattice.cubic(8.396),
        ["Fe", "Fe", "O"],
        [[0, 0, 0], [0.625, 0.625, 0.625], [0.380, 0.380, 0.380]],
    ).get_primitive_structure()


def _spins(structure: Structure, element: str) -> list[float]:
    return sorted(
        site.specie.spin for site in structure if site.specie.symbol == element
    )


def test_nio_lists_its_two_four_site_orderings_once_each() -> None:
    """A 2-site rock-salt cell orders only in a doubled cell. The doubled-cell
    structures repeat each pattern under several spin space groups and spin
    signs; reduced and deduplicated, two remain: AFM-II along [111] and the
    layered one along [001], both compensated."""
    orderings = enumerate_orderings(_rock_salt_nio(), ("Ni",), SYMPREC)

    assert [(len(o.structure), o.kind) for o in orderings] == [
        (4, "afm"),
        (4, "afm"),
    ]
    assert {o.structure.get_space_group_info()[0] for o in orderings} == {
        "R-3m",
        "P4/mmm",
    }
    for ordering in orderings:
        assert _spins(ordering.structure, "Ni") == [-1.0, 1.0]
        assert all(
            getattr(site.specie, "spin", None) is None
            for site in ordering.structure
            if site.specie.symbol == "O"
        )


def test_corundum_lists_the_three_compensated_patterns_of_its_cell() -> None:
    """The four Cr of corundum's primitive cell order three ways with two up
    and two down (Cr2O3's experimental +-+- among them); every listed
    ordering cancels by symmetry."""
    orderings = enumerate_orderings(_corundum_cr2o3(), ("Cr",), SYMPREC)

    in_cell = [o for o in orderings if len(o.structure) == 10]
    assert len(in_cell) == 3
    assert {o.kind for o in orderings} == {"afm"}
    assert all(len(o.structure) in (10, 20) for o in orderings)


def test_rutile_mnf2_lists_its_experimental_order_in_its_own_cell() -> None:
    """MnF2's corner and body-centre Mn antiparallel is the one ordering of
    its 6-site cell; the others need the doubled cell."""
    orderings = enumerate_orderings(_rutile_mnf2(), ("Mn",), SYMPREC)

    [own] = [o for o in orderings if len(o.structure) == 6]
    assert own.kind == "afm"
    assert _spins(own.structure, "Mn") == [-1.0, 1.0]


def test_spinel_orders_only_ferrimagnetically() -> None:
    """No operation maps a tetrahedral Fe of spinel onto an octahedral one,
    so no ordering of its six Fe cancels: all are ferrimagnetic, none is
    ferromagnetic."""
    orderings = enumerate_orderings(_spinel_fe3o4(), ("Fe",), SYMPREC)

    assert orderings
    assert {o.kind for o in orderings} == {"fim"}
    for ordering in orderings:
        assert set(_spins(ordering.structure, "Fe")) == {-1.0, 1.0}


@pytest.mark.parametrize(
    ("structure", "element"), [(_rutile_mnf2(), "Mn"), (_spinel_fe3o4(), "Fe")]
)
def test_the_kept_orderings_do_not_depend_on_spinforge_s_order_or_signs(
    structure: Structure, element: str
) -> None:
    """spinforge's moment basis has the signs the platform's LAPACK gives its
    singular vectors, so the same orderings can arrive in another order and
    orientation on another platform. Shuffled, with some reversed, they reduce
    to the same orderings, each with at least as many spin-up as spin-down
    sites."""
    raw = _raw_orderings(structure, (element,), SYMPREC)
    kept = _distinct(raw, (element,), SYMPREC)
    rng = random.Random(0)
    perturbed = [
        _reversed(ordering) if rng.random() < 0.5 else ordering
        for ordering in rng.sample(raw, len(raw))
    ]

    assert sorted(map(site_signature, _distinct(perturbed, (element,), SYMPREC))) == (
        sorted(map(site_signature, kept))
    )
    for ordering in kept:
        assert sum(_spins(ordering, element)) >= 0


def test_two_magnetic_elements_antiparallel_are_ferrimagnetic() -> None:
    """Rock-salt-like FeCoO2 with Fe and Co each on one sublattice: Fe up and
    Co down has equal spin counts but nothing maps Fe onto Co, so it is
    listed as ferrimagnetic, not antiferromagnetic."""
    structure = Structure(
        Lattice.cubic(4.2),
        ["Fe", "Fe", "Co", "Co", "O", "O", "O", "O"],
        [
            [0, 0, 0],
            [0.5, 0.5, 0],
            [0.5, 0, 0.5],
            [0, 0.5, 0.5],
            [0.5, 0, 0],
            [0, 0.5, 0],
            [0, 0, 0.5],
            [0.5, 0.5, 0.5],
        ],
    )

    orderings = enumerate_orderings(structure, ("Co", "Fe"), SYMPREC)

    sublattice_antiparallel = [
        o
        for o in orderings
        if len(set(_spins(o.structure, "Fe"))) == 1
        and len(set(_spins(o.structure, "Co"))) == 1
    ]
    assert sublattice_antiparallel
    assert {o.kind for o in sublattice_antiparallel} == {"fim"}


@pytest.mark.parametrize("element", ["Ni", "O"])
def test_only_the_listed_elements_carry_moments(
    element: str,
) -> None:
    """Only the listed elements carry moments: ordering O instead of Ni gives
    O moments and no Ni ones."""
    orderings = enumerate_orderings(_rock_salt_nio(), (element,), SYMPREC)

    other = "O" if element == "Ni" else "Ni"
    assert orderings
    for ordering in orderings:
        assert _spins(ordering.structure, element) == [-1.0, 1.0]
        assert all(
            getattr(site.specie, "spin", None) is None
            for site in ordering.structure
            if site.specie.symbol == other
        )
