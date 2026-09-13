from __future__ import annotations

import pytest
from pymatgen.core import Lattice, Structure

from goldilocks_core.analysis.symmetry import symmetry


def test_symmetry_resolves_space_group_for_a_simple_structure() -> None:
    silicon = Structure(Lattice.cubic(4.0), ["Si"], [[0.0, 0.0, 0.0]])

    state = symmetry(silicon)

    assert state.ok
    assert state.value.space_group_number > 0
    assert state.value.crystal_system == "cubic"


def test_symmetry_is_unavailable_for_disordered_structures() -> None:
    disordered = Structure(
        Lattice.cubic(4.0), [{"Fe": 0.5, "Ni": 0.5}], [[0.0, 0.0, 0.0]]
    )

    state = symmetry(disordered)

    assert not state.ok
    assert state.status == "unavailable"
    assert "disordered" in state.reason


def test_symmetry_analyzer_failure_degrades_to_unavailable_not_a_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v2 epic 4 (#1): the same failure v1 downgraded to None fields + a
    warning is now Unavailable -- named the same way geometry.py's
    equivalent failure now is too (goldilocks-core-design.md:299-310's A7
    asymmetry, fixed for both instead of just this one)."""
    silicon = Structure(Lattice.cubic(4.0), ["Si"], [[0.0, 0.0, 0.0]])

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise ValueError("synthetic spglib failure")

    monkeypatch.setattr("goldilocks_core.analysis.symmetry.SpacegroupAnalyzer", _boom)

    state = symmetry(silicon)

    assert not state.ok
    assert state.status == "unavailable"
    assert "synthetic spglib failure" in state.reason
