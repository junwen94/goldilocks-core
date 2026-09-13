from __future__ import annotations

import pytest
from pymatgen.core import Lattice, Structure

from goldilocks_core.analysis import (
    DimensionalityClassificationError,
    analyze_structure,
)


def test_disordered_structure_degrades_dimensionality_and_symmetry_gracefully() -> None:
    """A6: disordered structures (partial occupancy) make CrystalNN/spglib
    crash on a lot of real inputs, so analyze_structure special-cases
    ``not structure.is_ordered`` up front for both dimensionality and
    symmetry, before either external library ever runs — both degrade to a
    conservative default plus a warning, and the whole call still succeeds."""
    disordered = Structure(
        Lattice.cubic(4.0),
        [{"Fe": 0.5, "Ni": 0.5}],
        [[0.0, 0.0, 0.0]],
    )

    record = analyze_structure(disordered)

    assert record["dimensionality"] == "unknown"
    assert record["low_dimensional"] is False
    assert record["space_group_symbol"] is None
    assert record["space_group_number"] is None
    assert record["crystal_system"] is None
    assert any(
        "disordered" in warning.lower() for warning in record["analysis_warnings"]
    )


def test_dimensionality_classifier_failure_aborts_the_whole_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A7 (half): when CrystalNN/get_dimensionality_larsen raises on an
    *ordered* structure it cannot classify, analyze_structure re-raises as
    DimensionalityClassificationError and does not catch it — the entire
    analysis call fails, no record is produced at all. Contrast with the
    symmetry case below: an equivalent external-library failure there is
    caught and degrades a single field instead. See stfc/goldilocks-core#133."""
    silicon = Structure(Lattice.cubic(4.0), ["Si"], [[0.0, 0.0, 0.0]])

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise ValueError("synthetic CrystalNN failure")

    monkeypatch.setattr("goldilocks_core.analysis.get_dimensionality_larsen", _boom)

    with pytest.raises(DimensionalityClassificationError):
        analyze_structure(silicon)


def test_symmetry_analyzer_failure_degrades_one_field_instead_of_aborting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A7 (other half): the structurally identical failure — an external
    classifier raising on a valid-but-awkward *ordered* structure — is caught
    for symmetry and downgraded to None fields plus a warning, rather than
    aborting the call the way the dimensionality case above does. Both
    handled deliberately (see analysis.py's docstrings) but inconsistently;
    v2's tri-state design (goldilocks-core-design.md:299-310) exists to make
    both cases behave the same way instead of two different escape hatches."""
    silicon = Structure(Lattice.cubic(4.0), ["Si"], [[0.0, 0.0, 0.0]])

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise ValueError("synthetic spglib failure")

    monkeypatch.setattr("goldilocks_core.analysis.SpacegroupAnalyzer", _boom)

    record = analyze_structure(silicon)

    assert record["space_group_symbol"] is None
    assert record["space_group_number"] is None
    assert record["crystal_system"] is None
    assert any(
        "symmetry analysis failed" in warning.lower()
        for warning in record["analysis_warnings"]
    )
