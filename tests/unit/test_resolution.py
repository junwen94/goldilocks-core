from __future__ import annotations

import pytest
from pydantic import BaseModel

from goldilocks_core.inputs.overrides import HumanInput
from goldilocks_core.resolution import (
    Blocked,
    BlockedValueError,
    FieldState,
    Provenance,
    Resolved,
    ResolvedField,
    Unavailable,
)


def test_resolved_carries_value_and_source() -> None:
    state = Resolved(0.4, Provenance(source="heuristic"))

    assert state.ok is True
    assert state.status == "resolved"
    assert state.value == 0.4
    assert state.source == "heuristic"


def test_unavailable_is_not_ok_but_readable() -> None:
    state = Unavailable(reason="no ml model installed")

    assert state.ok is False
    assert state.status == "unavailable"
    assert state.reason == "no ml model installed"


def test_blocked_by_is_safe_but_every_other_attribute_poisons() -> None:
    state = Blocked(by="pseudopotential selection failed")

    assert state.ok is False
    assert state.status == "blocked"
    assert state.by == "pseudopotential selection failed"

    with pytest.raises(BlockedValueError, match="pseudopotential selection failed"):
        state.value  # noqa: B018 - deliberately accessing to trigger the poison


def test_blocked_chain_walks_to_the_original_root_cause() -> None:
    root = Blocked(by="Ce has no fully-relativistic pseudopotential")
    downstream = Blocked(by=root)
    further_downstream = Blocked(by=downstream)

    assert downstream.root_cause() == "Ce has no fully-relativistic pseudopotential"
    assert further_downstream.root_cause() == (
        "Ce has no fully-relativistic pseudopotential"
    )


def _double_if_resolved(state: FieldState[float]) -> FieldState[float]:
    """A toy advisor-shaped function following the mandatory hand-written
    propagation pattern (goldilocks-core-design.md S4.2 SS2): check ``.ok``
    first, return ``Blocked`` if not, otherwise compute normally."""
    if not state.ok:
        return Blocked(by=state)
    return Resolved(state.value * 2, state.provenance)


def _double_without_checking(state: FieldState[float]) -> FieldState[float]:
    """The bug this whole design exists to catch: forgot the ``.ok`` check."""
    return Resolved(state.value * 2, state.provenance)


class TestPoisonObjectPropagation:
    """The design doc's own mandatory test: swap any advisor's upstream for
    ``Blocked``, it must either return ``Blocked`` or raise
    ``BlockedValueError`` -- never a normal value."""

    def test_a_correct_advisor_propagates_blocked(self) -> None:
        upstream = Blocked(by="pseudopotential selection failed")

        result = _double_if_resolved(upstream)

        assert isinstance(result, Blocked)
        assert result.root_cause() == "pseudopotential selection failed"

    def test_a_buggy_advisor_that_skips_the_check_crashes_instead_of_computing(
        self,
    ) -> None:
        upstream = Blocked(by="pseudopotential selection failed")

        with pytest.raises(BlockedValueError):
            _double_without_checking(upstream)

    def test_a_correct_advisor_still_computes_on_a_resolved_upstream(self) -> None:
        upstream = Resolved(3.0, Provenance(source="heuristic"))

        result = _double_if_resolved(upstream)

        assert isinstance(result, Resolved)
        assert result.value == 6.0


class TestResolvedFieldBoundaryProjection:
    """The day-one blocker's actual answer: settings/facts classes never hold FieldState
    directly."""

    def test_resolved_projects_to_a_plain_serializable_shape(self) -> None:
        projection = ResolvedField[float].from_state(
            Resolved(0.4, Provenance(source="heuristic"))
        )

        assert projection.model_dump() == {
            "status": "resolved",
            "value": 0.4,
            "source": "heuristic",
            "reason": None,
            "blocked_by": None,
        }

    def test_unavailable_projects_with_a_reason_and_no_value(self) -> None:
        projection = ResolvedField[float].from_state(
            Unavailable(reason="no ml model installed")
        )

        assert projection.status == "unavailable"
        assert projection.value is None
        assert projection.reason == "no ml model installed"

    def test_blocked_projects_with_blocked_by_and_no_value(self) -> None:
        projection = ResolvedField[float].from_state(
            Blocked(by="pseudopotential selection failed")
        )

        assert projection.status == "blocked"
        assert projection.value is None
        assert projection.blocked_by == "pseudopotential selection failed"

    def test_projection_is_json_serializable(self) -> None:
        projection = ResolvedField[float].from_state(
            Resolved(0.4, Provenance(source="heuristic"))
        )

        assert projection.model_dump_json()


# --- Worked example: contract declaration -> resolution -> pydantic serialization ---
# Lives here, not in src/, because it exists to prove resolution.py's design works
# inside a real settings-shaped class, not to be a real advisor -- no consumer exists
# for a "convergence settings" contract until v2 epic 6 builds advisors/convergence.py.

HEURISTIC_MIXING_BETA = 0.4
"""v1's ``DEFAULT_MIXING_BETA`` (``advice/parameters.py``), unchanged."""


class _ConvergenceHumanInput(HumanInput):
    mixing_beta: float | None = None


def _resolve_mixing_beta(human: _ConvergenceHumanInput) -> FieldState[float]:
    """The source-axis pattern (goldilocks-core-design.md S4.2): one
    if/elif per tier, highest priority first. ml/llm are stubbed to "not
    available yet" rather than omitted, so this starts from a real
    4-branch shape instead of silently forgetting a tier -- matching how
    v2 epics 4-6 stub the ml branch everywhere until v2 epic 11 wires it
    up for real."""
    if human.mixing_beta is not None:
        return Resolved(
            human.mixing_beta,
            Provenance(source="human", source_note="operator override"),
        )
    ml_value: float | None = None  # no ml model for mixing_beta; stubbed until epic 11
    if ml_value is not None:
        return Resolved(ml_value, Provenance(source="ml"))
    llm_value: float | None = None  # no agent wiring yet; stubbed until epic 11
    if llm_value is not None:
        return Resolved(llm_value, Provenance(source="llm"))
    return Resolved(
        HEURISTIC_MIXING_BETA,
        Provenance(source="heuristic", source_note="package default"),
    )


class _ConvergenceSettings(BaseModel):
    mixing_beta: ResolvedField[float]

    @classmethod
    def from_human_input(cls, human: _ConvergenceHumanInput) -> _ConvergenceSettings:
        return cls(mixing_beta=ResolvedField.from_state(_resolve_mixing_beta(human)))


class TestWorkedExample:
    def test_human_override_resolves_and_serializes_end_to_end(self) -> None:
        settings = _ConvergenceSettings.from_human_input(
            _ConvergenceHumanInput(mixing_beta=0.25)
        )

        assert settings.mixing_beta.status == "resolved"
        assert settings.mixing_beta.value == 0.25
        assert settings.mixing_beta.source == "human"
        assert settings.model_dump_json()
        assert "mixing_beta" in _ConvergenceSettings.model_json_schema()["properties"]

    def test_falls_back_to_the_heuristic_default(self) -> None:
        settings = _ConvergenceSettings.from_human_input(_ConvergenceHumanInput())

        assert settings.mixing_beta.status == "resolved"
        assert settings.mixing_beta.value == 0.4
        assert settings.mixing_beta.source == "heuristic"

    def test_blocked_upstream_produces_a_blocked_settings_field(self) -> None:
        settings = _ConvergenceSettings(
            mixing_beta=ResolvedField.from_state(
                Blocked(by="pseudopotential selection failed")
            )
        )

        assert settings.mixing_beta.status == "blocked"
        assert settings.mixing_beta.value is None
        assert settings.mixing_beta.blocked_by == "pseudopotential selection failed"
        # still an ordinary pydantic model even when every field is blocked -- this is
        # the whole point of storing the projection, not the poison object, on the class
        assert settings.model_dump_json()
