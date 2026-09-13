from __future__ import annotations

from goldilocks_core.advisors.boundary import boundary
from goldilocks_core.analysis.geometry import GeometryFacts
from goldilocks_core.resolution import Blocked, Provenance, Resolved


def _geometry(dimensionality: str, low_dimensional: bool):
    return Resolved(
        GeometryFacts(dimensionality=dimensionality, low_dimensional=low_dimensional),
        Provenance(source="heuristic"),
    )


def test_bulk_3d_gets_no_isolation_correction() -> None:
    state = boundary(_geometry("3d", False))

    assert state.ok
    assert state.value.assume_isolated == "none"
    assert state.value.warnings == ()


def test_molecule_gets_martyna_tuckerman() -> None:
    """A8: a molecule in a periodic box gets a spurious image-interaction
    offset without a real-space Coulomb cutoff."""
    state = boundary(_geometry("molecule", True))

    assert state.value.assume_isolated == "martyna-tuckerman"


def test_1d_wire_gets_martyna_tuckerman() -> None:
    state = boundary(_geometry("1d", True))

    assert state.value.assume_isolated == "martyna-tuckerman"


def test_2d_slab_gets_a_warning_instead_of_a_guess() -> None:
    """A9 (asymmetric-slab dipole correction) is deferred, not guessed at
    here -- a slab's image-interaction fix depends on which direction is
    the vacuum gap and whether the slab is symmetric, which this fact
    doesn't have."""
    state = boundary(_geometry("2d", True))

    assert state.value.assume_isolated == "none"
    assert len(state.value.warnings) == 1
    assert "dipole correction" in state.value.warnings[0]


def test_blocked_geometry_propagates() -> None:
    state = boundary(Blocked(by="synthetic failure"))

    assert not state.ok
    assert state.status == "blocked"
    assert state.root_cause() == "synthetic failure"
