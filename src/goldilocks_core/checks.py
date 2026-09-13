"""checks: the one place in the whole system allowed to hard-fail.

New in v2 (v2 epic 6, #6); no v1 precedent. Per
goldilocks-core-design.md's own summary (line 4007): "``check_all`` is
the sole gateway for cross-parameter constraints, and also the boundary
between ``advise`` and ``generate``." Every advisor in this codebase
degrades field-by-field (``Resolved``/``Unavailable``/``Blocked``) and
never raises; a runnable input is only refused here, by returning a
non-empty ``CheckReport.blocking`` for a future ``generate()`` to read
-- not by raising an exception. **By convention, any other ``raise`` in
generation/advisor code found after this epic should be treated as
suspicious** (the issue's own acceptance criterion), since this module
is now where that decision belongs.

**Two independent mechanisms, not one:**

1. ``collect_blocked`` -- cross-parameter constraints are not the only
   way to end up unable to generate a runnable input: any upstream
   advisor that itself resolved to ``Blocked`` should also end up in
   ``report.blocking``, with its root cause surfaced (the design doc's
   own "blocking is derivable... the rejection reason follows the
   ``blocked_by`` chain back to the root cause, no separate
   error-message system needed", goldilocks-core-design.md:3261-3263).
   This mechanism is generic: it takes any ``FieldState`` from any
   advisor, not just the ones this epic happens to build.

2. A named cross-parameter rule -- ``occupations='fixed'`` together with
   a spin-polarized system requires an *integer* ``tot_magnetization``
   (QE's own requirement once occupation numbers are no longer
   automatically split by smearing). This is scoped to the ``scf``
   step only: an ``nscf`` step reads the charge/spin density a prior
   ``scf`` step already converged and does not re-derive this
   constraint (goldilocks-core-design.md:3417, "per-step constraint").
   ``advisors/occupations.py`` already flags this combination with an
   informational warning when it produces ``fixed``, but only *this*
   module owns the actual numeric check and the decision to block.

**Everything else in the design doc's "constraint check table"**
(memory overflow, disordered-structure generation blocking, the
tri-tier override-conflict rules) needs infrastructure this epic does
not build -- HPC resource estimates, a real orchestrator assembling
``system_settings``/``per_step`` -- and is deliberately left as a gap
for whichever future epic adds it, not attempted here. This module's
job is to establish *that* single gate exists and works for what is
checkable today, not to enumerate every rule up front.
"""

from __future__ import annotations

from dataclasses import dataclass

from goldilocks_core.advisors.magnetic_config import MagneticConfigFacts
from goldilocks_core.advisors.occupations import OccupationsDecision
from goldilocks_core.resolution import Blocked, FieldState

_MISSING_TOT_MAGNETIZATION = (
    "occupations='fixed' with a spin-polarized system requires an explicit "
    "integer tot_magnetization, but none was set."
)


@dataclass(frozen=True, slots=True)
class CheckReport:
    blocking: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.blocking


def collect_blocked(*field_states: FieldState[object]) -> tuple[str, ...]:
    """Every ``Blocked`` field state's root cause, in the order given.

    ``Unavailable`` and ``Resolved`` states are not blocking on their
    own -- an advisor that could not tell, or a caller that never
    supplied a given field at all, is not the same as an upstream
    failure. Only genuine ``Blocked`` propagation reaches here.
    """
    return tuple(
        state.root_cause() for state in field_states if isinstance(state, Blocked)
    )


def check_all(
    *field_states: FieldState[object],
    occupations: FieldState[OccupationsDecision] | None = None,
    magnetic: FieldState[MagneticConfigFacts] | None = None,
    purpose: str = "scf",
) -> CheckReport:
    """The single hard-fail gate.

    ``field_states`` is every ``FieldState`` a caller has assembled so
    far (from any advisor) -- any that are ``Blocked`` land in
    ``report.blocking`` via ``collect_blocked``. ``occupations``/
    ``magnetic``/``purpose`` additionally opt into the one named
    cross-parameter rule this epic implements; passing them here also
    folds them into the generic ``Blocked`` scan, so a caller does not
    need to repeat them in ``field_states`` as well.
    """
    optional = tuple(s for s in (occupations, magnetic) if s is not None)
    blocking = list(collect_blocked(*field_states, *optional))

    reason = _fixed_occupations_needs_integer_moment(occupations, magnetic, purpose)
    if reason is not None:
        blocking.append(reason)

    return CheckReport(blocking=tuple(blocking))


def _fixed_occupations_needs_integer_moment(
    occupations: FieldState[OccupationsDecision] | None,
    magnetic: FieldState[MagneticConfigFacts] | None,
    purpose: str,
) -> str | None:
    if purpose != "scf" or occupations is None or magnetic is None:
        return None
    if not (occupations.ok and magnetic.ok):
        return None  # already surfaced via collect_blocked, or genuinely unknown
    if occupations.value.occupations != "fixed" or not magnetic.value.spin_polarized:
        return None
    tot = magnetic.value.tot_magnetization
    if tot is None:
        return _MISSING_TOT_MAGNETIZATION
    if tot != int(tot):
        return (
            "occupations='fixed' with a spin-polarized system requires an "
            f"integer tot_magnetization; got {tot!r}."
        )
    return None
