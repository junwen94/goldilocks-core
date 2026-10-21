from __future__ import annotations

from support import geometry_facts

from goldilocks_core.advisors.vdw_method import (
    VdwMethodHumanInput,
    VdwMethodLlmInput,
    vdw_method,
)
from goldilocks_core.resolution import Blocked, Provenance, Resolved


def _geometry(dimensionality: str):
    return Resolved(geometry_facts(dimensionality), Provenance(source="heuristic"))


def test_bulk_3d_gets_no_vdw_correction() -> None:
    state = vdw_method(_geometry("3d"), "PBEsol")

    assert state.ok
    assert state.value.use_vdw is False
    assert state.value.method is None


def test_low_dimensional_structure_defaults_to_d3bj() -> None:
    state = vdw_method(_geometry("2d"), "PBEsol")

    assert state.value.use_vdw is True
    assert state.value.method == "d3bj"


def test_vdw_inclusive_functional_skips_the_correction_even_when_low_dimensional() -> (
    None
):
    state = vdw_method(_geometry("2d"), "vdW-DF")

    assert state.value.use_vdw is False
    assert state.value.method is None
    assert state.value.warnings == ()


def test_vdw_inclusive_functional_warns_if_a_method_was_explicitly_requested() -> None:
    state = vdw_method(
        _geometry("2d"),
        "vdW-DF",
        human=VdwMethodHumanInput(method="d3bj"),
    )

    # human.use_vdw is still None here, so the vdw-inclusive-functional guard
    # runs -- but human.method being set should still surface a warning.
    assert state.value.use_vdw is False
    assert len(state.value.warnings) == 1
    assert "double-counting" in state.value.warnings[0].message
    assert state.value.warnings[0].code == "vdw.double_counting_avoided"


def test_human_override_wins_over_heuristic() -> None:
    state = vdw_method(
        _geometry("3d"), "PBEsol", human=VdwMethodHumanInput(use_vdw=True)
    )

    assert state.value.use_vdw is True
    assert state.value.method == "d3bj"
    assert state.source == "human"


def test_llm_override_is_used_when_no_human_override_is_given() -> None:
    state = vdw_method(_geometry("2d"), "PBEsol", llm=VdwMethodLlmInput(use_vdw=False))

    assert state.value.use_vdw is False
    assert state.source == "llm"


def test_blocked_geometry_propagates() -> None:
    state = vdw_method(Blocked(by="synthetic failure"), "PBEsol")

    assert not state.ok
    assert state.status == "blocked"
    assert state.root_cause() == "synthetic failure"
