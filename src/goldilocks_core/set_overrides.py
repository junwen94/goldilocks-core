"""set_overrides: the shared request-validation path (v2 epic 8, #8;
goldilocks-core-design.md S3577-3584 P3/P5, S4.2's `--set` sections).

One path, three callers: CLI's `--set KEY=VALUE`, an HTTP request body's
override object, and an MCP tool call's override argument all end up
here, building the same `RunOverrides` `service.advise()`/`generate()`
already take -- directly answers this epic's own evidence #1 ("backend
-option validation is CLI-only"): there is now exactly one place that
knows which keys exist and what they mean, not one per transport.

**Not a hand-written key list**: every key this module accepts comes
from `capabilities.bindings()`, which is `capabilities()`'s own
settings-schema walk projected to construction recipes instead of JSON
-- add an advisor's override field and it is immediately both
`--set`-able here and visible in `capabilities()`, from the same single
walk (`capabilities.py`'s own docstring explains why one walk, not two).

**CLI vs HTTP/MCP split**: `coerce_cli_value`/`parse_set_flags` are
CLI-only, converting shell strings into typed Python values (booleans,
numbers, JSON for array/object-shaped settings like `k_grid`/
`u_by_element`). HTTP/MCP request bodies are JSON already, so their
values reach `build_overrides` pre-typed and skip coercion entirely --
`build_overrides` itself is the one transport-agnostic core, matching
P1/P5: validate identically, fail the same way, regardless of caller.

**`steps.<name>.<key>=value` is accepted but not yet load-bearing**:
`service`'s own per-step advisors compute exactly one `StepAdvice` per
request today (`service/_step.py`'s own docstring: no `PlannedStep`/
task-expansion layer exists, since only one task/step,
`scf_single_point`, is wired). The `steps.` prefix here is validated
against `_STEP_NAMES` and then stripped -- functionally identical to
the flat key today, because there is nowhere else for a per-step
override to land. Design point 2's hard rule ("a flat `--set` landing
on >1 step must warn") has no real trigger case until a second step
exists; implementing that warning now would be inventing behaviour
with no way to test it against a real multi-step task, the same
"don't build a lookup table for one entry" reasoning `inputs/task.py`
and `steps.py` already document for this exact codebase.
"""

from __future__ import annotations

import difflib
import json

from pydantic import ValidationError

from goldilocks_core.capabilities import SettingBinding, bindings
from goldilocks_core.failures import ExpectedFailure
from goldilocks_core.service import (
    KpointsOverrides,
    ResourceOverrides,
    RunOverrides,
    StepOverrides,
    SystemOverrides,
)

_STEP_NAMES = ("scf",)
"""The one step epics 1-7 built a generation writer for -- see this
module's own docstring on why `steps.<name>.` is validated against
this but does not yet route anywhere different from a flat key."""

_TRUE_STRINGS = {"true", "1", "yes", "on"}
_FALSE_STRINGS = {"false", "0", "no", "off"}


class InvalidSetting(ExpectedFailure, ValueError):
    """A `--set`/override key doesn't exist, or its value doesn't match
    what that key's setting declares. Carries a message already safe
    and specific enough to show a user directly (P6's user-error path)."""

    kind = "invalid_setting"


def _did_you_mean(key: str, candidates: list[str]) -> str:
    matches = difflib.get_close_matches(key, candidates, n=3)
    if matches:
        return f"unknown setting {key!r}; did you mean: {', '.join(matches)}?"
    return f"unknown setting {key!r}; run 'goldilocks settings' to see valid keys"


def parse_set_flags(raw: list[str]) -> dict[str, str]:
    """CLI-only: splits repeated `--set KEY=VALUE` flags into a flat
    dict, handling `steps.<name>.<key>=value` by validating the step
    name and stripping the prefix -- see this module's own docstring on
    why that prefix is a no-op today. Raises `InvalidSetting` for
    malformed flags (no `=`, empty key, or the same key given twice
    with two different values) -- P5's "validate immediately after
    parsing" applied to `--set` itself, not just backend options.
    """
    assignments: dict[str, str] = {}
    for item in raw:
        if "=" not in item:
            raise InvalidSetting(f"--set value {item!r} is not KEY=VALUE")
        raw_key, _, value = item.partition("=")
        key = _resolve_step_prefix(raw_key.strip())
        if not key:
            raise InvalidSetting(f"--set value {item!r} has an empty key")
        if key in assignments and assignments[key] != value:
            raise InvalidSetting(
                f"--set {key!r} given twice with different values "
                f"({assignments[key]!r} and {value!r})"
            )
        assignments[key] = value
    return assignments


def _resolve_step_prefix(key: str) -> str:
    if not key.startswith("steps."):
        return key
    parts = key.split(".", 2)
    if len(parts) != 3:
        raise InvalidSetting(f"malformed steps.<name>.<key> setting: {key!r}")
    _steps, step_name, inner_key = parts
    if step_name not in _STEP_NAMES:
        available = ", ".join(_STEP_NAMES)
        raise InvalidSetting(
            f"unknown step {step_name!r} in {key!r}; available steps: {available}"
        )
    return inner_key


def coerce_cli_value(binding: SettingBinding, key: str, raw: str) -> object:
    """Convert one `--set KEY=VALUE` flag's raw string into the type
    `binding.json_type` declares. array/object-shaped settings
    (`k_grid`, `u_by_element`, ...) are parsed as JSON, since a shell
    string has no native array/object syntax of its own; everything
    else uses a direct, minimal rule."""
    kind = binding.json_type["type"]
    if kind == "boolean":
        lowered = raw.strip().lower()
        if lowered in _TRUE_STRINGS:
            return True
        if lowered in _FALSE_STRINGS:
            return False
        raise InvalidSetting(f"{key}={raw!r} is not a boolean (true/false)")
    if kind == "integer":
        try:
            return int(raw)
        except ValueError:
            raise InvalidSetting(f"{key}={raw!r} is not an integer") from None
    if kind == "number":
        try:
            return float(raw)
        except ValueError:
            raise InvalidSetting(f"{key}={raw!r} is not a number") from None
    if kind in ("array", "object"):
        try:
            return json.loads(raw)
        except json.JSONDecodeError as error:
            raise InvalidSetting(f"{key}={raw!r} is not valid JSON: {error}") from None
    return raw


def coerce_cli_assignments(raw_assignments: dict[str, str]) -> dict[str, object]:
    """`parse_set_flags`'s output, typed -- looks up each key's binding
    (raising `InvalidSetting` with a did-you-mean suggestion for an
    unknown one, P3's spirit) and coerces its string value."""
    catalogue = bindings()
    typed: dict[str, object] = {}
    for key, raw in raw_assignments.items():
        binding = catalogue.get(key)
        if binding is None:
            raise InvalidSetting(_did_you_mean(key, list(catalogue)))
        typed[key] = coerce_cli_value(binding, key, raw)
    return typed


def build_overrides(assignments: dict[str, object]) -> RunOverrides:
    """The transport-agnostic core: `assignments` values must already
    be Python-typed (bool/int/float/str/list/dict) -- an HTTP/MCP JSON
    body already is; a CLI caller must run `coerce_cli_assignments`
    first. Unknown keys raise `InvalidSetting` with a did-you-mean
    suggestion; a value that doesn't fit its setting's real pydantic
    contract (wrong enum member, wrong shape) raises `InvalidSetting`
    with that contract's own validation message."""
    catalogue = bindings()
    grouped: dict[tuple[str, str], dict[str, object]] = {}
    outer_cls: dict[tuple[str, str], type | None] = {}
    for key, value in assignments.items():
        binding = catalogue.get(key)
        if binding is None:
            raise InvalidSetting(_did_you_mean(key, list(catalogue)))
        slot = (binding.branch, binding.outer_field)
        outer_cls[slot] = binding.human_input_cls
        field_name = binding.inner_field or binding.outer_field
        grouped.setdefault(slot, {})[field_name] = value

    branch_fields: dict[str, dict[str, object]] = {
        "system": {},
        "kpoints": {},
        "resources": {},
    }
    for (branch, outer_field), field_values in grouped.items():
        cls = outer_cls[(branch, outer_field)]
        if cls is None:
            branch_fields[branch][outer_field] = field_values[outer_field]
            continue
        try:
            branch_fields[branch][outer_field] = cls(**field_values)
        except ValidationError as error:
            raise InvalidSetting(
                f"invalid value(s) for {outer_field!r}: {error}"
            ) from error

    return RunOverrides(
        system=SystemOverrides(**branch_fields["system"]),
        step=StepOverrides(
            kpoints=KpointsOverrides(**branch_fields["kpoints"]),
            resources=ResourceOverrides(**branch_fields["resources"]),
        ),
    )


def build_overrides_from_cli(raw: list[str]) -> RunOverrides:
    """The full CLI path: `--set` strings in, a ready `RunOverrides` out."""
    return build_overrides(coerce_cli_assignments(parse_set_flags(raw)))
