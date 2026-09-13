from __future__ import annotations

import pytest
from pymatgen.core import Lattice, Structure

from goldilocks_core.analysis.geometry import geometry


def test_geometry_resolves_3d_for_a_bulk_structure() -> None:
    # A single primitive-cell Si-Si pair is too far apart for JmolNN's
    # cutoff table to bond at all; a real diamond-cubic cell (8 atoms,
    # generated from the space group) has genuine nearest-neighbour bonds.
    silicon = Structure.from_spacegroup(
        "Fd-3m", Lattice.cubic(5.431), ["Si"], [[0.0, 0.0, 0.0]]
    )

    state = geometry(silicon)

    assert state.ok
    assert state.value.dimensionality == "3d"
    assert state.value.low_dimensional is False


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
