"""convergence: SCF convergence thresholds, scaled by system size.

New in v2 (v2 epic 6, #6). Last of five per-step advisors
(``occupations -> k_sampling -> n_irr_k -> nbnd -> convergence``,
goldilocks-core-design.md:3239-3248).

**`conv_thr`/`etot_conv_thr` scale with `nat`.** Replaces v1's
``_advise_convergence`` (``advice/parameters.py:276-306``), which took
only ``hints`` and always returned a fixed ``DEFAULT_CONV_THR = 1e-6``
regardless of system size -- QE's own docs concede ``conv_thr`` is
extensive, like the total energy, so a fixed value is too loose for a
large system and possibly too tight for a small one, with no scaling
formula supplied. This adopts aiida-quantumespresso's actual
``PwBaseWorkChain`` implementation instead
(``workflows/pw/base.py:271-272``, goldilocks-core-design.md:2914):
``conv_thr = nat * conv_thr_per_atom`` and
``etot_conv_thr = nat * etot_conv_thr_per_atom``, using its
production-validated balanced-tier constants
(``conv_thr_per_atom=0.2e-9``, ``etot_conv_thr_per_atom=1e-5``) rather
than invented ones. ``mixing_beta``/``electron_maxstep`` keep v1's own
flat defaults (``0.4``, ``80`` -- ``advice/parameters.py:17-18``); no
source in this codebase scales those with system size.

**`mixing_mode='local-TF'` for 2d/molecule geometries**
(goldilocks-core-design.md:2907, "F26"): QE's default ``'plain'``
mixing struggles on structures with a vacuum region (the long-range
part of Thomas-Fermi screening mixing handles better); ``'1d'`` is
deliberately *not* included -- the design doc's finding names only
``2d``/``molecule``, and this module does not extend that without a
source.

**`mixing_fixed_ns=50` when +U is active**
(goldilocks-core-design.md:2908, "F5"): an engineering heuristic, not a
literature-standardized constant -- freezing the occupation-matrix
mixing for the first N iterations helps a Hubbard-corrected SCF avoid
oscillating between metastable occupation-matrix solutions early on.
The design doc's own finding notes a QE-mailing-list two-stage restart
procedure as the fallback when 50 iterations are not enough, but
choosing to run that restart is a multi-step workflow decision, not
something a single stateless call can decide -- out of scope here.

**No `is_metal` input.** It appears in the design doc's own illustrative
per-step loop snippet, but no source in this codebase specifies a
metallicity-dependent convergence rule (e.g. a different `mixing_beta`
for metals) -- unlike `k_sampling`'s deliberately-unused `occupations`
input, there is no documented future contract this would need to match,
so it is left out rather than carried as an unjustified placeholder.

``needs_correlation``/``geometry`` are optional: anything other than a
confirmed ``Resolved`` value (``Unavailable``, ``Blocked``, or simply
not supplied) just skips the corresponding optional adjustment --
``conv_thr``/``etot_conv_thr``/``mixing_beta``/``electron_maxstep`` do
not depend on either, so this module never blocks on them.

**``Provenance.field_sources``, populated here first.** The one
``Provenance`` on this whole ``ConvergenceDecision`` answers "did a
human/llm touch *anything* in here" (``source`` above); v2 epic 9
(#9)'s scenario 2 needs the finer "did a human touch *this specific*
field" (overriding ``mixing_beta`` alone must not make ``conv_thr``/
``mixing_mode`` look human-sourced too) -- ``field_sources`` carries
that, one entry per field, using the same priority order as ``_pick``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from goldilocks_core.analysis.geometry import GeometryFacts
from goldilocks_core.inputs.overrides import HumanInput, LlmInput
from goldilocks_core.resolution import FieldState, Provenance, Resolved, Warning

MixingMode = Literal["plain", "local-TF"]

_CONV_THR_PER_ATOM = 0.2e-9
_ETOT_CONV_THR_PER_ATOM = 1e-5
_DEFAULT_MIXING_BETA = 0.4
"""Matches v1's DEFAULT_MIXING_BETA (advice/parameters.py:17)."""
_DEFAULT_ELECTRON_MAXSTEP = 80
"""Matches v1's DEFAULT_ELECTRON_MAXSTEP (advice/parameters.py:18)."""
_HUBBARD_MIXING_FIXED_NS = 50
_LOCAL_TF_DIMENSIONALITIES = frozenset({"2d", "molecule"})


@dataclass(frozen=True, slots=True)
class ConvergenceDecision:
    conv_thr: float
    etot_conv_thr: float
    mixing_beta: float
    electron_maxstep: int
    mixing_mode: MixingMode
    mixing_fixed_ns: int | None
    warnings: tuple[Warning, ...] = ()


class ConvergenceHumanInput(HumanInput):
    conv_thr: float | None = None
    etot_conv_thr: float | None = None
    mixing_beta: float | None = None
    electron_maxstep: int | None = None
    mixing_mode: MixingMode | None = None
    mixing_fixed_ns: int | None = None


class ConvergenceLlmInput(LlmInput):
    mixing_beta: float | None = None
    electron_maxstep: int | None = None
    mixing_mode: MixingMode | None = None
    mixing_fixed_ns: int | None = None


def convergence(
    nat: int,
    needs_correlation: FieldState[bool] | None = None,
    geometry: FieldState[GeometryFacts] | None = None,
    human: ConvergenceHumanInput | None = None,
    llm: ConvergenceLlmInput | None = None,
) -> FieldState[ConvergenceDecision]:
    human = human or ConvergenceHumanInput()
    llm = llm or ConvergenceLlmInput()

    hubbard_active = bool(
        needs_correlation is not None
        and needs_correlation.ok
        and needs_correlation.value
    )
    local_tf = bool(
        geometry is not None
        and geometry.ok
        and geometry.value.dimensionality in _LOCAL_TF_DIMENSIONALITIES
    )

    conv_thr = (
        human.conv_thr if human.conv_thr is not None else nat * _CONV_THR_PER_ATOM
    )
    decision = ConvergenceDecision(
        conv_thr=conv_thr,
        etot_conv_thr=(
            human.etot_conv_thr
            if human.etot_conv_thr is not None
            else nat * _ETOT_CONV_THR_PER_ATOM
        ),
        mixing_beta=_pick(human.mixing_beta, llm.mixing_beta, _DEFAULT_MIXING_BETA),
        electron_maxstep=_pick(
            human.electron_maxstep, llm.electron_maxstep, _DEFAULT_ELECTRON_MAXSTEP
        ),
        mixing_mode=_pick(
            human.mixing_mode, llm.mixing_mode, "local-TF" if local_tf else "plain"
        ),
        mixing_fixed_ns=_pick(
            human.mixing_fixed_ns,
            llm.mixing_fixed_ns,
            _HUBBARD_MIXING_FIXED_NS if hubbard_active else None,
        ),
    )
    field_sources = {
        "conv_thr": _field_source(human.conv_thr),
        "etot_conv_thr": _field_source(human.etot_conv_thr),
        "mixing_beta": _field_source(human.mixing_beta, llm.mixing_beta),
        "electron_maxstep": _field_source(human.electron_maxstep, llm.electron_maxstep),
        "mixing_mode": _field_source(human.mixing_mode, llm.mixing_mode),
        "mixing_fixed_ns": _field_source(human.mixing_fixed_ns, llm.mixing_fixed_ns),
    }
    source = "human" if _any_set(human) else "llm" if _any_set(llm) else "heuristic"
    return Resolved(decision, Provenance(source=source, field_sources=field_sources))


def _pick[T](human_value: T | None, llm_value: T | None, default: T) -> T:
    if human_value is not None:
        return human_value
    if llm_value is not None:
        return llm_value
    return default


def _field_source(human_value: object | None, llm_value: object | None = None) -> str:
    """Per-scalar counterpart to ``_pick``'s priority order -- used to
    populate ``Provenance.field_sources`` (v2 epic 9, #9, scenario 2)
    without changing what value each field itself resolves to."""
    if human_value is not None:
        return "human"
    if llm_value is not None:
        return "llm"
    return "heuristic"


def _any_set(overrides: HumanInput) -> bool:
    return bool(overrides.model_dump(exclude_defaults=True))
