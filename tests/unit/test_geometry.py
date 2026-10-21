from __future__ import annotations

import pytest
from pymatgen.core import Lattice, Structure

from goldilocks_core.analysis.geometry import geometry, vacuum_axes


def test_geometry_resolves_3d_for_a_bulk_structure() -> None:
    # A real diamond-cubic cell (8 atoms, generated from the space group),
    # whose covalent Si-Si bonds JmolNN's cutoffs already find. A 1-atom
    # cubic Si cell would rank 3d too, bonded to its own images through
    # the nearest-neighbour shell, but it is not a real crystal.
    silicon = Structure.from_spacegroup(
        "Fd-3m", Lattice.cubic(5.431), ["Si"], [[0.0, 0.0, 0.0]]
    )

    state = geometry(silicon)

    assert state.ok
    assert state.value.dimensionality == "3d"
    assert state.value.low_dimensional is False


def _graphene(c: float) -> Structure:
    """A graphene sheet (C-C 1.42 A) in a cell ``c`` A tall."""
    return Structure(
        Lattice.hexagonal(2.46, c), ["C", "C"], [[0, 0, 0.5], [1 / 3, 2 / 3, 0.5]]
    )


def test_a_monolayer_has_vacuum_along_c_only() -> None:
    """The widest atom-free slab along a is 2/3 of the 2.13 A spacing of
    the (100) planes; along c it is the whole 15 A cell."""
    monolayer = _graphene(15.0)

    state = geometry(monolayer)

    assert state.value.vacuum_axes == ("c",)
    assert state.value.vacuum_widths_angstrom == (1.42, 1.42, 15.0)
    assert vacuum_axes(monolayer) == ("c",)


def test_a_molecule_in_a_box_has_vacuum_along_every_axis() -> None:
    """An Si2 dimer (bond 2.25 A, along [111]) in a 12 A cubic box leaves
    10.7 A of vacuum along each lattice vector."""
    offset = 2.25 / (12.0 * 3**0.5)
    dimer = Structure(
        Lattice.cubic(12.0), ["Si", "Si"], [[0, 0, 0], [offset, offset, offset]]
    )

    state = geometry(dimer)

    assert state.value.vacuum_axes == ("a", "b", "c")
    assert state.value.vacuum_widths_angstrom == (10.7, 10.7, 10.7)


def test_a_bulk_crystal_has_no_vacuum() -> None:
    silicon = Structure.from_spacegroup(
        "Fd-3m", Lattice.cubic(5.431), ["Si"], [[0.0, 0.0, 0.0]]
    )

    state = geometry(silicon)

    assert state.value.vacuum_axes == ()
    assert state.value.vacuum_widths_angstrom == (1.36, 1.36, 1.36)


def test_unwrapped_coordinates_are_wrapped_into_the_cell() -> None:
    """AB-stacked graphene sheets 3.35 A apart in a 20 A cell, written with
    fractional z of -0.084 and 1.084 as an unwrapped trajectory frame would:
    the sheets sit at z = 1.675 A and 18.325 A, with 16.65 A of vacuum
    between them."""
    z = 3.35 / 40
    bilayer = Structure(
        Lattice.hexagonal(2.46, 20.0),
        ["C", "C", "C", "C"],
        [[0, 0, -z], [1 / 3, 2 / 3, -z], [0, 0, 1 + z], [2 / 3, 1 / 3, 1 + z]],
    )

    assert geometry(bilayer).value.vacuum_widths_angstrom[2] == 16.65
    assert vacuum_axes(bilayer) == ("c",)


@pytest.mark.parametrize(("c", "axes"), [(4.99, ()), (5.0, ()), (5.01, ("c",))])
def test_a_gap_counts_as_vacuum_only_beyond_5_angstrom(
    c: float, axes: tuple[str, ...]
) -> None:
    assert vacuum_axes(_graphene(c)) == axes


def test_geometry_is_unavailable_for_disordered_structures() -> None:
    disordered = Structure(
        Lattice.cubic(4.0), [{"Fe": 0.5, "Ni": 0.5}], [[0.0, 0.0, 0.0]]
    )

    state = geometry(disordered)

    assert not state.ok
    assert state.status == "unavailable"
    assert "disordered" in state.reason


def test_dimensionality_classifier_failure_degrades_to_unavailable_not_a_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v2 epic 4 (#1), closing stfc/goldilocks-core#133: v1's equivalent
    failure here raised DimensionalityClassificationError uncaught and
    aborted the whole analyze_structure call, unlike the symmetry case
    (goldilocks-core-design.md:299-310's A7 asymmetry). Now both are
    Unavailable the same way -- and geometry.py no longer even depends on
    CrystalNN, the thing that used to fail here."""
    silicon = Structure(Lattice.cubic(4.0), ["Si"], [[0.0, 0.0, 0.0]])

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise ValueError("synthetic dimensionality failure")

    monkeypatch.setattr(
        "goldilocks_core.analysis.geometry.get_dimensionality_larsen", _boom
    )

    state = geometry(silicon)

    assert not state.ok
    assert state.status == "unavailable"
    assert "synthetic dimensionality failure" in state.reason
