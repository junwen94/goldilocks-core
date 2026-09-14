"""A6/A7 pitfall regression, v2 shape (v2 epic 9, #9).

v1's ``analyze_structure`` handled a classifier failure inconsistently:
a disordered structure or a dimensionality-classifier failure aborted
the whole call (uncaught exception), while an equivalent symmetry
-classifier failure degraded just that one field to ``None`` plus a
warning. v2's tri-state design (``analysis/geometry.py``,
``analysis/symmetry.py``) makes both cases behave the same way --
``Unavailable``, never a raise -- which is what these tests now confirm
directly against those two functions instead of v1's single combined
``analyze_structure`` entry point (deleted alongside the rest of v1's
tree at the end of this epic).
"""

from __future__ import annotations

from unittest.mock import patch

from pymatgen.core import Lattice, Structure

from goldilocks_core.analysis.geometry import geometry
from goldilocks_core.analysis.symmetry import symmetry


def test_disordered_structure_degrades_dimensionality_and_symmetry_gracefully() -> None:
    """A6: disordered structures (partial occupancy) make CrystalNN/spglib
    crash on a lot of real inputs, so both geometry() and symmetry()
    special-case ``not structure.is_ordered`` up front, before either
    external library ever runs -- both degrade to ``Unavailable`` with a
    reason naming why, and neither call raises."""
    disordered = Structure(
        Lattice.cubic(4.0),
        [{"Fe": 0.5, "Ni": 0.5}],
        [[0.0, 0.0, 0.0]],
    )

    dimensionality = geometry(disordered)
    space_group = symmetry(disordered)

    assert not dimensionality.ok
    assert "disordered" in dimensionality.reason
    assert not space_group.ok
    assert "disordered" in space_group.reason


def test_dimensionality_classifier_failure_degrades_instead_of_aborting() -> None:
    """A7 (half): v1's analyze_structure re-raised as
    DimensionalityClassificationError when CrystalNN/get_dimensionality_
    larsen raised on an *ordered* structure it could not classify --
    aborting the entire analysis, no record produced at all. v2's
    geometry() catches the same failure and degrades to Unavailable
    instead, matching symmetry()'s own handling below rather than being a
    second, inconsistent escape hatch. See stfc/goldilocks-core#133."""
    silicon = Structure(Lattice.cubic(4.0), ["Si"], [[0.0, 0.0, 0.0]])

    with patch(
        "goldilocks_core.analysis.geometry.get_dimensionality_larsen",
        side_effect=ValueError("synthetic CrystalNN failure"),
    ):
        dimensionality = geometry(silicon)

    assert not dimensionality.ok
    assert "dimensionality classification failed" in dimensionality.reason


def test_symmetry_analyzer_failure_degrades_one_field_instead_of_aborting() -> None:
    """A7 (other half): the structurally identical failure -- an external
    classifier raising on a valid-but-awkward *ordered* structure -- was
    already caught and downgraded for symmetry in v1; v2 keeps that
    behavior (Unavailable, not a raise), now consistent with geometry()
    above instead of the odd one out."""
    silicon = Structure(Lattice.cubic(4.0), ["Si"], [[0.0, 0.0, 0.0]])

    with patch(
        "goldilocks_core.analysis.symmetry.SpacegroupAnalyzer",
        side_effect=ValueError("synthetic spglib failure"),
    ):
        space_group = symmetry(silicon)

    assert not space_group.ok
    assert "symmetry analysis failed" in space_group.reason
