"""The v2 field-resolution primitive: three independent axes per field.

    source axis   human > ml > llm > heuristic     which knowledge the
                                                    value came from
    scope axis    per-step > global                which steps the
                                                    value governs
    status axis   resolved / unavailable / blocked whether the value
                                                    exists at all

This module owns the source and status axes (``Provenance``/``Source``
and ``Resolved``/``Unavailable``/``Blocked``); the scope axis has no
code yet, it shows up once per-step settings exist (v2 epic 6).

Kept as one top-level module, not a package: ``Provenance`` needs a
fresh v2-only shape (``Source`` is the clean four-tier list, not v1's
six-value, two-concerns-conflated ``ProvenanceSource``) and can't reuse
v1's existing ``goldilocks_core.provenance`` module without breaking
the still-running v1 code that constructs the old shape -- but a type
that every ``Resolved`` field must carry is exactly the kind of
genuinely cross-cutting primitive that belongs at top level, next to
``kmesh.py``/``functionals.py``/``units.py``, not inside any one
``analysis/``/``advisors/`` module (goldilocks-core-design.md S13:
three named top-level files, no ``utils/``). v1's ``provenance.py`` is
left untouched; it still describes v1's own advisors until they're
deleted (v2 epic 9).

``Resolved``/``Unavailable``/``Blocked`` are core's *internal*
representation, used inside analysis/ and advisors/. They are
deliberately not pydantic models and never appear directly on a
settings/facts class: goldilocks-core-design.md's "day-one blocker"
section requires those classes to store the serializable *projection*
(``ResolvedField``, below) instead. This is not v1's abandoned
dual-projection design come back (``to_jsonable``/``to_portable`` were
two projections of *one* type) -- it is two type *families*, the same
technique already used for "public records vs local handles".

No ``contracts/`` package here, on purpose: v1 already tried
centralizing every record into one hub (``goldilocks_core.contracts``,
deleted 2026-09-02, "records and serialization merge into the
behavioral modules that own them") and reversed it after it caused
import cycles and organizational drift -- see the ruff ``banned-api``
entry for ``goldilocks_core.contracts``. The v2 design doc's proposed
``contracts/`` tree didn't know about that history when it was
written; per-domain contract classes (e.g. ``analysis/is_metal.py``
defining its own Fact model) is the resolution, decided with the user
2026-09-14. Only ``Provenance``/``Source`` and the tri-state itself are
exempt, for the reason above: they have no single owning domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

from pydantic import BaseModel

Status = Literal["resolved", "unavailable", "blocked"]

Source = Literal["human", "ml", "llm", "heuristic"]
"""Order is priority, highest first: a human-set value always wins
over one an ml model, llm, or heuristic would have chosen; likewise ml
over llm over heuristic."""

WarningLevel = Literal["info", "warning", "error"]


class Warning(BaseModel):
    """A machine-actionable warning an advisor's decision can carry
    alongside its value (goldilocks-agent-design.md: the tool-response
    ``warnings`` array must be structured, not prose-only, so an agent
    can relay/filter/group it without parsing text).

    ``code`` is a stable identifier (``"<record>.<slug>"``, e.g.
    ``"job.walltime_defaulted"``) an advisor commits not to rename once
    shipped -- callers may match on it. ``category`` is the same record
    name used throughout ``Advice.records()``/``capabilities()``
    (``"job"``, ``"hubbard"``, ...): reusing that existing, already
    -stable vocabulary instead of inventing a second one a frontend
    would need a separate legend for. ``message`` is the human-readable
    text, with any per-occurrence values (an element, a number) already
    interpolated in -- the whole reason this array exists is so a
    caller never has to parse it to know what happened, but the text is
    still there for direct display."""

    code: str
    level: WarningLevel
    category: str
    message: str


class Provenance(BaseModel):
    source: Source
    data_source: str | None = None
    """Which upstream record/dataset backed the value, when ``source``
    is ``"ml"`` -- e.g. a contract version or artifact id. ``None`` for
    tiers with nothing to cite."""
    model_id: str | None = None
    """Which model produced the value, when ``source`` is ``"ml"``.
    ``None`` otherwise."""
    source_note: str | None = None
    """Free-text detail v1's split ``ProvenanceSource`` values used to
    carry structurally (e.g. "derived from structure analysis" vs
    "package default, nothing else applied") -- now just prose here
    instead of a family of source literals."""


class BlockedValueError(Exception):
    """Raised by touching any attribute of a ``Blocked`` field other
    than ``by``/``ok``/``status``. Getting this exception where a value
    was expected means a caller forgot to check ``.ok`` before using a
    field -- the whole point of the poison-object design is that this
    is a loud, immediate crash instead of a silently-wrong number."""


@dataclass(frozen=True, slots=True)
class Resolved[T]:
    """A field that has a real value. Falling back to a lower source
    tier (e.g. no ml model installed, heuristic decided instead) is
    still ``Resolved`` -- source and status are independent axes;
    degrading tiers is not the same thing as not having a value."""

    value: T
    provenance: Provenance
    ok: ClassVar[bool] = True
    status: ClassVar[Status] = "resolved"

    @property
    def source(self) -> str:
        return self.provenance.source


@dataclass(frozen=True, slots=True)
class Unavailable:
    """A field genuinely not determinable after trying every source
    tier. Not an error: ``advise()`` must still return a result with
    this field explicitly unavailable and why, per the "diagnosis is
    always available" promise."""

    reason: str
    ok: ClassVar[bool] = False
    status: ClassVar[Status] = "unavailable"


@dataclass(frozen=True, slots=True)
class Blocked:
    """A field that cannot even be computed because something it
    depends on already failed. ``by`` is the upstream ``FieldState``
    (usually another ``Blocked``) that caused this, so walking ``.by``
    repeatedly reaches the root cause without a separate error-message
    system -- see ``root_cause``.

    Any attribute access other than ``by``/``ok``/``status`` raises
    ``BlockedValueError`` instead of returning a normal-looking value.
    This is deliberate: propagation is hand-written
    (``if not x.ok: return Blocked(by=x)``), not automatic, because
    automatic propagation is a framework with no cheap fallback for its
    own bugs. The poison object is that fallback for the hand-written
    version -- forgetting a check crashes immediately instead of
    computing a silently-wrong number.
    """

    by: Blocked | str
    ok: ClassVar[bool] = False
    status: ClassVar[Status] = "blocked"

    def __getattr__(self, name: str) -> None:
        raise BlockedValueError(
            f"{self.root_cause()!r} is missing, {name!r} is not available"
        )

    def root_cause(self) -> str:
        """Walk the ``by`` chain to the original string reason."""
        cause = self.by
        while isinstance(cause, Blocked):
            cause = cause.by
        return cause


type FieldState[T] = Resolved[T] | Unavailable | Blocked
"""``Resolved[T] | Unavailable | Blocked``. Written out as a plain
union rather than a generic alias class: matches the design doc's own
notation and keeps ``isinstance``/``match`` working directly on the
three real classes above, with no extra indirection."""


def blocked_by(state: Unavailable | Blocked) -> Blocked | str:
    """Turn a failed upstream ``FieldState`` into a value ``Blocked.by``
    accepts, for functions that depend on another analysis/advisors fact
    rather than only on raw input (v2 epic 4, #1's ``needs_soc`` reading
    ``composition`` is the first real case). ``Blocked.by``'s type is
    ``Blocked | str``, not ``FieldState`` -- passing an ``Unavailable``
    straight through would silently break ``root_cause()``'s walk, since it
    stops at the first non-``Blocked`` value and returns it as-is, whatever
    it is. This function is the one place that distinction gets handled:
    an ``Unavailable``'s ``reason`` becomes the string cause, and a
    ``Blocked`` passes through unchanged so the chain stays walkable."""
    if isinstance(state, Blocked):
        return state
    return state.reason


class ResolvedField[T](BaseModel):
    """The serializable projection of a ``FieldState[T]``, for storage
    on a settings/facts class -- this answers the design doc's
    "day-one blocker" directly: a contract field's declared type is
    ``ResolvedField[SomeType]``, never ``FieldState[SomeType]``.
    Because this is an ordinary pydantic model, ``model_dump()``,
    OpenAPI schema generation, and the ``capabilities`` auto-generation
    all see a plain, already-supported shape -- never the poison
    object. Build one with ``ResolvedField.from_state``, not the
    constructor directly, so the fields below can't be filled in an
    inconsistent combination (e.g. a ``status`` of ``"blocked"`` next
    to a non-``None`` ``value``)."""

    status: Status
    value: T | None = None
    source: str | None = None
    reason: str | None = None
    blocked_by: str | None = None

    @classmethod
    def from_state(cls, state: FieldState[T]) -> ResolvedField[T]:
        if isinstance(state, Resolved):
            return cls(status=state.status, value=state.value, source=state.source)
        if isinstance(state, Unavailable):
            return cls(status=state.status, reason=state.reason)
        return cls(status=state.status, blocked_by=state.root_cause())
