from __future__ import annotations

import pytest
from support import geometry_facts

from goldilocks_core.advisors.boundary import (
    BoundaryHumanInput,
    BoundaryLlmInput,
    boundary,
)
from goldilocks_core.resolution import Blocked, Provenance, Resolved


def _geometry(dimensionality: str, vacuum_axes: tuple[str, ...] = ()):
    return Resolved(
        geometry_facts(dimensionality, vacuum_axes), Provenance(source="heuristic")
    )


def test_bulk_3d_gets_no_isolation_correction() -> None:
    state = boundary(_geometry("3d"))

    assert state.ok
    assert state.value.assume_isolated == "none"
    assert state.value.warnings == ()


@pytest.mark.parametrize(
    ("dimensionality", "vacuum_axes"),
    [
        ("molecule", ()),
        ("1d", ()),
        ("1d", ("a", "b")),
        ("2d", ("c",)),
        ("molecule", ("a", "b")),
    ],
)
def test_a_cell_periodic_along_any_axis_gets_no_isolation_correction(
    dimensionality: str, vacuum_axes: tuple[str, ...]
) -> None:
    """#126: a molecular crystal (CO2 ranks 'molecule' but fills its cell),
    a wire or a slab is periodic along at least one lattice vector, and
    pw.x has no 1D truncation, so Martyna-Tuckerman would replace real
    periodic electrostatics with those of an isolated cluster."""
    state = boundary(_geometry(dimensionality, vacuum_axes))

    assert state.value.assume_isolated == "none"


@pytest.mark.parametrize("dimensionality", ["molecule", "3d"])
def test_vacuum_along_all_three_axes_gets_martyna_tuckerman(
    dimensionality: str,
) -> None:
    """A8: a molecule or cluster in a box picks up a spurious interaction
    with its own periodic images. The vacuum decides, not the bond rank: a
    lone atom bonded to its images 5-6 A away ranks '3d'."""
    state = boundary(_geometry(dimensionality, ("a", "b", "c")))

    assert state.value.assume_isolated == "martyna-tuckerman"
    assert state.source == "heuristic"
    assert state.value.warnings == ()


def test_2d_slab_gets_a_warning_instead_of_a_guess() -> None:
    """A9 (asymmetric-slab dipole correction) is deferred, not guessed at
    here -- a slab's image-interaction fix depends on which direction is
    the vacuum gap and whether the slab is symmetric, which this fact
    doesn't have."""
    state = boundary(_geometry("2d", ("c",)))

    assert state.value.assume_isolated == "none"
    assert len(state.value.warnings) == 1
    assert "dipole correction" in state.value.warnings[0].message
    assert state.value.warnings[0].code == "boundary.dipole_correction_suggested"


def test_blocked_geometry_propagates() -> None:
    state = boundary(Blocked(by="synthetic failure"))

    assert not state.ok
    assert state.status == "blocked"
    assert state.root_cause() == "synthetic failure"


def test_human_override_bypasses_a_blocked_geometry() -> None:
    """Regression for #36 (v2 epic 9, #9): boundary() used to have no
    override path at all, unlike its sibling vdw_method (same epic,
    same GeometryFacts input) -- a Blocked/Unavailable geometry (e.g. a
    disordered structure dimensionality classification can't handle)
    unconditionally blocked the whole run via this one field, with no
    escape hatch, even if the caller supplied every other override."""
    state = boundary(
        Blocked(by="synthetic failure"),
        human=BoundaryHumanInput(assume_isolated="martyna-tuckerman"),
    )

    assert state.ok
    assert state.value.assume_isolated == "martyna-tuckerman"
    assert state.source == "human"


def test_llm_override_bypasses_a_blocked_geometry_too() -> None:
    state = boundary(
        Blocked(by="synthetic failure"),
        llm=BoundaryLlmInput(assume_isolated="none"),
    )

    assert state.ok
    assert state.source == "llm"
