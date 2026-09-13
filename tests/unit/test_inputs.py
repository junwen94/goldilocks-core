from __future__ import annotations

import pytest
from pydantic import ValidationError

from goldilocks_core.inputs import HumanInput, LlmInput, Task


class _ExampleHumanInput(HumanInput):
    mixing_beta: float | None = None


def test_human_input_rejects_unknown_fields_instead_of_silently_ignoring_them() -> None:
    with pytest.raises(ValidationError):
        _ExampleHumanInput(mixing_beta=0.25, typo_field=1)


def test_llm_input_has_the_identical_shape_to_human_input_by_construction() -> None:
    assert issubclass(LlmInput, HumanInput)
    assert LlmInput.model_fields.keys() == HumanInput.model_fields.keys()


def test_task_replaces_calculation_intent_with_independent_fields() -> None:
    task = Task()

    assert task.code == "quantum_espresso"
    assert task.task == "scf_single_point"
