"""StepSettings: the typed middle layer between a planned step and a
rendered one.

New in v2 (v2 epic 6, #6); no v1 precedent. Per
goldilocks-core-design.md:1054-1253 ("generation: task decides the step
sequence"), a fully rendered ``Step`` is not enough by itself: a DOS
task's ``scf`` and ``nscf`` steps differ in grid, ``nbnd``, occupations,
and convergence threshold, and leaving that up to the writer to
improvise is exactly what "generation only translates" is supposed to
forbid. ``StepSettings`` is the resolved, ready-to-render answer for one
step -- by the time one of these is constructed, ``checks.py`` (this
epic's other deliverable) has already confirmed nothing it depends on
is ``Blocked``, so these dataclasses hold concrete values, not
``FieldState``-wrapped ones.

**Typed per program, not one class with unused fields.** An earlier
shape considered giving every step every field, leaving whatever a
given program does not use as ``None`` -- rejected
(goldilocks-core-design.md:1133-1150) because it both force-feeds
``dos.x`` fields it cannot use (``k_sampling``/``n_irr_k``/``nbnd``/
``convergence``) and has nowhere to put what ``dos.x`` actually needs
(``emin``/``delta_e``/``ngauss``). ``StepSettings`` is instead a tagged
union (``PwSettings | DosSettings | PhSettings``), dispatched by
ordinary ``isinstance``/pattern matching -- there is no shared base
data class for the union members, only ``BaseStepSettings`` below,
which really is common to every program.

**Only the fields this codebase currently has a producer for are
required.** ``PwSettings``'s ``occupations``/``k_sampling``/``n_irr_k``/
``nbnd``/``convergence`` come directly from this epic's five per-step
advisors (``advisors/occupations.py`` etc.) and are required. Every
other field on every dataclass here -- ``BaseStepSettings``'s
``disk_io``/``size``/``mem_per_proc`` (resource sizing, v2 epic 7),
``PwSettings``'s ``parallel``/``relax`` (a parallelization advisor and
relax/vc-relax support, v2 epic 10), and all of ``DosSettings``'s and
``PhSettings``'s fields (their own per-step advisors do not exist yet)
-- is declared per the design doc's shape but defaults to ``None``,
deliberately: nothing in this codebase computes them yet, and forcing
every construction site (including this epic's own tests) to fabricate
a placeholder value for a field nothing produces would be worse than an
honest, typed "not yet available". These defaults are expected to
become required once each field's real producer exists -- a natural,
signature-breaking change at that point, not scope creep now.

``DosSettings`` existing as its own type with its own ``broadening``
field (QE calls this ``degauss`` too, in ``dos.x``/``projwfc.x`` --
occupation broadening in ``pw.x`` and DOS-histogram broadening in
``dos.x`` are physically unrelated numbers that happen to share a QE
keyword) is exactly what stops one value from silently doing double
duty across two different programs -- the deferred test from v2 epic 2
this epic's issue names as its item 6.
"""

from __future__ import annotations

from dataclasses import dataclass

from goldilocks_core.advisors.convergence import ConvergenceDecision
from goldilocks_core.advisors.k_sampling import KSamplingDecision
from goldilocks_core.advisors.nbnd import NbndDecision
from goldilocks_core.advisors.occupations import OccupationsDecision


@dataclass(frozen=True, slots=True)
class BaseStepSettings:
    """Common to every program: disk-usage convention and per-process
    resource estimates. No producer in this codebase yet -- v2 epic 7's
    "resource sizing" -- so every field defaults to unknown."""

    disk_io: str | None = None
    size: dict[str, float] | None = None
    mem_per_proc: float | None = None


@dataclass(frozen=True, slots=True)
class PwSettings(BaseStepSettings):
    """pw.x: scf / nscf / bands / relax."""

    occupations: OccupationsDecision | None = None
    k_sampling: KSamplingDecision | None = None
    n_irr_k: int | None = None
    nbnd: NbndDecision | None = None
    convergence: ConvergenceDecision | None = None
    parallel: dict[str, int] | None = None
    relax: dict[str, float] | None = None


@dataclass(frozen=True, slots=True)
class DosSettings(BaseStepSettings):
    """dos.x / projwfc.x: reads a previous step's wavefunctions --
    no k_sampling/n_irr_k/nbnd/convergence of its own."""

    emin: float | None = None
    emax: float | None = None
    delta_e: float | None = None
    broadening: float | None = None
    ngauss: int | None = None


@dataclass(frozen=True, slots=True)
class PhSettings(BaseStepSettings):
    """ph.x: phonon q-point sampling and self-consistency threshold."""

    q_sampling: dict[str, float] | None = None
    tr2_ph: float | None = None
    epsil: bool | None = None
    parallel: dict[str, int] | None = None


type StepSettings = PwSettings | DosSettings | PhSettings
