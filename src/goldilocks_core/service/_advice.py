"""``Advice``/``RunOverrides``: the pipeline-wide shapes composing the
three phases in ``_analysis.py``/``_system.py``/``_step.py`` (v2 epic 8,
#8).

Composed of the three phases' own dataclasses, not a flat 20-odd-field
dataclass: a flat shape would need to import every analysis/advisor
decision type directly for its own field annotations, which is exactly
the "god module" shape ``scripts/check_complexity.py``'s import-surface
ceiling exists to catch. Composing three already-typed sub-objects costs
this module only three local origins.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

from pydantic import BaseModel
from pymatgen.core import Structure

from goldilocks_core.resolution import FieldState, Resolved
from goldilocks_core.service._analysis import AnalysisFacts, AnalysisOverrides
from goldilocks_core.service._step import StepAdvice, StepOverrides
from goldilocks_core.service._system import SystemAdvice, SystemOverrides


@dataclass(frozen=True, slots=True)
class RunOverrides:
    """One optional human/llm override object per advisor this epic
    wires up, grouped by the same three phases as ``Advice`` below.
    Deliberately typed, not a generic key/value bag -- ``resolution.py``'s
    own note applies here too: "there is no generic override-forwarding
    mechanism; each advisor's override class is separate and typed."
    Mapping flat ``--set``/HTTP keys onto these fields is the shared
    request-validation layer's job (this epic's next module), not this
    one's."""

    analysis: AnalysisOverrides = field(default_factory=AnalysisOverrides)
    system: SystemOverrides = field(default_factory=SystemOverrides)
    step: StepOverrides = field(default_factory=StepOverrides)


@dataclass(frozen=True, slots=True)
class Advice:
    """Every ``FieldState`` (plus the two plain, non-tri-state resource
    values from ``advisors/size.py``, held on ``step``) produced for one
    structure/code/task request. ``checks.check_all(*advice.field_states(),
    ...)`` decides whether ``generate()`` may proceed; ``advice.records()``
    is what ``BundleInput.records`` should carry unchanged -- tri-state,
    not the unwrapped values ``SystemSettings``/``PwSettings`` need."""

    structure: Structure
    analysis: AnalysisFacts
    system: SystemAdvice
    step: StepAdvice

    def field_states(self) -> tuple[FieldState[object], ...]:
        """Every ``FieldState`` except ``occupations``/``magnetic``,
        which ``checks.check_all`` folds in itself -- see its own
        docstring on why a caller must not repeat them here."""
        return (
            self.analysis.field_states()
            + self.system.field_states()
            + self.step.field_states()
        )

    def records(self) -> dict[str, FieldState[object]]:
        """One entry per analysis/advisor decision, keyed by the same
        name used throughout this package -- the manifest's ``records``
        object. No such key set is codified anywhere else in the
        codebase; this is epic 8's own convention.

        Every value is passed through ``_json_safe_state`` first:
        advisor decisions are plain (non-pydantic) frozen dataclasses --
        ``bundle.py``'s ``ResolvedField``/``bundle_files`` never had to
        serialize one of these to JSON before, since no orchestrator fed
        it real advisor output until this epic. ``dataclasses.asdict``
        handles the general case (including nested dataclasses like
        ``PseudoTable.asset``, and tuples of them like
        ``pseudopotentials``); the one exception is
        ``MagneticConfigFacts.relabeled_structure``, a pymatgen
        ``Structure`` (not JSON-encodable and, today, always identical
        to the input structure already implicit in the bundle -- see
        that field's own docstring), which is dropped rather than
        converted.
        """
        merged = {
            **self.analysis.records(),
            **self.system.records(),
            **self.step.records(),
        }
        return {name: _json_safe_state(state) for name, state in merged.items()}

    def warnings(self) -> list[dict[str, object]]:
        """Every advisor decision's own ``resolution.Warning`` tuple,
        flattened into one machine-actionable list -- goldilocks-agent
        -design.md's "tool return value must carry a warnings array,
        the agent must relay it verbatim" requirement. Each entry is
        already a plain ``{"code", "level", "category", "message"}``
        dict (``_json_safe_value`` below converts every ``Warning``
        instance on the way into ``records()``), so a caller can
        filter/group by ``code``/``category`` without parsing text --
        the actual, literal ask, not just prose relayed verbatim."""
        collected: list[dict[str, object]] = []
        for _name, state in sorted(self.records().items()):
            if isinstance(state, Resolved) and isinstance(state.value, dict):
                collected.extend(state.value.get("warnings") or ())
        return collected


def _json_safe_value(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        data = dataclasses.asdict(value)
        data.pop("relabeled_structure", None)
        return _json_safe_value(data)
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, dict):
        return {key: _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    return value


def _json_safe_state(state: FieldState[object]) -> FieldState[object]:
    if isinstance(state, Resolved):
        return Resolved(_json_safe_value(state.value), state.provenance)
    return state
