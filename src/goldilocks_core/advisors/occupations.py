"""occupations: fixed | smearing(type, degauss), from is_metal, the spin
mode and the electron count.

New in v2 (v2 epic 6, #6). The first of five per-step advisors
(`occupations -> k_sampling -> n_irr_k -> nbnd -> convergence`,
goldilocks-core-design.md:3239-3248) -- ``k_sampling`` takes this
module's ``degauss`` as an *explicit* input rather than assuming a fixed
sigma the way v1's dataset generation implicitly did.

Replaces v1's ``_advise_smearing`` (``advice/parameters.py:101-155``),
which branched on v1's more granular ``analysis["electronic_character"]``
(``"metal"``/``"likely_metal"``/``"insulator"``/unknown -- a
v1-only classification that no longer exists in v2;
``analysis/is_metal.py``'s ``FieldState[Metallicity]`` replaces it, more
conservatively: ``Unavailable`` rather than a guessed ``"likely_metal"``).

Metals get QE's cold smearing (``degauss=0.01`` Ry, matching v1's own
``METALLIC_SMEARING_WIDTH_RY``, ``advice/parameters.py:19``); a
*confirmed* non-metal gets ``fixed`` occupations where they can run
(below, #138), smearing otherwise. ``is_metal``
``Unavailable`` -- composition alone could not confirm metallic
character either way -- also gets smearing, not fixed. This is a
deliberate departure from v1's own mapping (v1's "unknown" bucket also
fell through to fixed): v1's "unknown" was genuinely no-signal-at-all,
but v2's ``is_metal`` is more conservative than v1's
``electronic_character`` (``analysis/is_metal.py``'s own docstring: it
only ever confidently asserts "metal", never a guessed "non_metal"), so
its ``Unavailable`` bucket is broader and now includes cases that look
metallic but are not fully confirmed. Smearing is the safe universal
default there: it tolerates a small or absent gap, whereas fixed
occupations assumes one and can fail to converge, or converge to the
wrong state, on an actual metal it misclassified. This also matches
common practice elsewhere (e.g. aiida-quantumespresso's own protocols
default to smearing rather than requiring a prior metal/insulator
classification).

``tetrahedra_opt`` (QE's third occupations option, appearing in
goldilocks-core-design.md's per-step pipeline sketch) is not decided by
this heuristic: no source in this codebase specifies when it should win
over smearing/fixed, so this module does not guess. It remains a valid
override value -- ``human``/``llm`` can request it directly -- and a
task protocol can give it to one of its steps (``OccupationsProtocol``,
the dos task's nscf step, #174), just not something the ``is_metal`` rule
will ever choose on its own. A protocol ranks below a request's
``occupations`` and above the llm; its record reads ``heuristic`` with
the protocol named in ``source_note``. A request's ``smearing_type``/
``degauss`` apply under it only where it chooses smearing.

**Why ``magnetic`` and ``electron_count`` are inputs (#138).** One QE
constraint couples the occupations, the moment and the electron count:
fixed occupations fill nelup = (nelec + tot_magnetization)/2 and neldw =
(nelec - tot_magnetization)/2 bands, so a collinear spin-polarized run
(``MagneticConfigFacts.spin_mode`` ``collinear``, nspin=2) needs an
integer ``tot_magnetization`` of nelec's parity -- pw.x stops without one
(QE 7.3 ``PW/src/input.f90:764-778``) -- and a spin-unpolarized one an
even nelec, since nspin=1 fills ``NINT(nelec)/2`` doubly occupied bands
(``PW/src/iweights.f90``). This module owns that constraint, so no
default is one ``checks.py`` refuses. ``OccupationsDecision
.tot_magnetization`` is the moment the step pins and the only one the
writer renders: the human request admitted by ``magnetic_config`` on
every collinear run, plus, under fixed occupations only, 0 for an applied
antiferromagnetic ordering that is compensated by symmetry
(``MagneticConfigFacts.compensated``) and whose electron count is known and
even. That zero holds by construction under fixed occupations; on
smearing it would pin a moment nobody asked for.

A heuristic non-metal therefore gets ``fixed`` only where fixed
occupations can run: always for a spin-polarized noncollinear
(spin-orbit) run, which takes no ``tot_magnetization``, breaks time
reversal and fills single-electron spinor bands (#146); for a run without
spin polarization, spin-orbit or not, unless nelec is odd -- without
moments spin-orbit coupling keeps time reversal, so every band has a
degenerate Kramers partner and an odd count half-fills a pair; for a
collinear run only with a pinned integer moment of nelec's parity.
Otherwise it gets
smearing. An explicit ``fixed`` stays fixed, with the derived moment
where there is one; a collinear run without one carries
``FIXED_SPIN_POLARIZED_WARNING``, and ``checks.py`` refuses it. An
electron count that is not resolved (a table not installed yet, a normal
preview) never blocks this decision -- ``nbnd`` and generation block on
it already -- it only skips the parity test, and the derived 0 then
waits for the count. A magnetic decision that is not resolved blocks the
input on its own; this module then reads no spin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from goldilocks_core.advisors.electron_count import ElectronCountDecision
from goldilocks_core.advisors.magnetic_config import MagneticConfigFacts, SpinMode
from goldilocks_core.analysis.is_metal import Metallicity
from goldilocks_core.inputs.overrides import HumanInput, LlmInput
from goldilocks_core.resolution import (
    Blocked,
    FieldState,
    Provenance,
    Resolved,
    Warning,
    field_sources_from,
)
from goldilocks_core.types import SmearingType

Occupations = Literal["fixed", "smearing", "tetrahedra_opt"]

_METALLIC_DEGAUSS = 0.01
"""Ry. Matches v1's METALLIC_SMEARING_WIDTH_RY (advice/parameters.py:19)."""

_SPIN_CHANNEL_TOLERANCE = 1e-6
"""Matches QE's own ``eps8``-scale tolerance (``PW/src/setup.f90``'s
odd-electron check) for treating a summed ``nelec`` as whole despite
floating-point noise, without treating a genuinely fractional one (a
partial-occupancy structure) as whole."""

FIXED_SPIN_POLARIZED_WARNING = Warning(
    code="occupations.fixed_spin_polarized_needs_integer_tot_magnetization",
    level="warning",
    category="occupations",
    message=(
        "occupations='fixed' on a collinear spin-polarized run needs an integer "
        "tot_magnetization, and none was set or derived; the input is refused "
        "until tot_magnetization is set or occupations is smearing."
    ),
)

WARNING_CATALOGUE = (FIXED_SPIN_POLARIZED_WARNING,)
"""Every warning code this module can emit -- ``capabilities.py``'s
``warnings[]`` catalogue aggregates one of these tuples per advisor."""


@dataclass(frozen=True, slots=True)
class OccupationsDecision:
    """``tot_magnetization`` is the moment this step pins, the only one
    the writer renders (module docstring); ``None`` pins none."""

    occupations: Occupations
    smearing_type: SmearingType | None
    degauss: float | None
    tot_magnetization: float | None = None
    warnings: tuple[Warning, ...] = ()


def whole_spin_channels(nelec: float, tot_magnetization: float) -> bool:
    """Whether nelup = (nelec + tot_magnetization)/2 and neldw =
    (nelec - tot_magnetization)/2 are both whole numbers, as fixed
    occupations need (#70); with ``tot_magnetization`` 0, whether nelec
    is even."""
    return all(
        abs(round(count) - count) <= _SPIN_CHANNEL_TOLERANCE
        for count in (
            (nelec + tot_magnetization) / 2,
            (nelec - tot_magnetization) / 2,
        )
    )


@dataclass(frozen=True, slots=True)
class _Spin:
    """What the occupations choice reads of the run's spin: its mode,
    whether it is spin-polarized, the admitted human moment, whether the
    applied ordering is compensated by symmetry, the electron count when
    known, and whether these spin facts are a resolved magnetic record's
    (``from_magnetic``; without one the run reads as spin-unpolarized)."""

    mode: SpinMode
    polarized: bool
    requested: float | None
    compensated: bool
    nelec: float | None
    from_magnetic: bool

    @property
    def fixed_moment(self) -> float | None:
        """The ``tot_magnetization`` fixed occupations pin on a collinear
        run: the human request, else 0 for a compensated ordering with a
        known even electron count, else ``None``."""
        if self.mode != "collinear":
            return None
        if self.requested is not None:
            return self.requested
        if self.compensated and self.nelec is not None:
            return 0.0 if whole_spin_channels(self.nelec, 0.0) else None
        return None

    @property
    def fixed_runs(self) -> bool:
        """Whether fixed occupations can run (module docstring); an
        unknown electron count skips the parity test."""
        if self.mode == "noncollinear" and self.polarized:
            return True
        moment = self.fixed_moment if self.polarized else 0.0
        if moment is None or moment != int(moment):
            return False
        return self.nelec is None or whole_spin_channels(self.nelec, moment)

    @property
    def fixed_runs_records(self) -> tuple[str, ...]:
        """The records whose facts ``fixed_runs`` reads (#173): ``magnetic``
        when its spin facts are that record's, and ``electron_count`` when
        it tests the count's parity (``counts_parity``)."""
        records = ("magnetic",) if self.from_magnetic else ()
        return (*records, "electron_count") if self.counts_parity else records

    @property
    def counts_parity(self) -> bool:
        """Whether ``fixed_runs`` tests a known electron count's parity:
        against 0 without spin polarization, a requested integer moment, or
        the 0 of a compensated collinear ordering. A spin-polarized
        noncollinear run needs no count, and a collinear one with no such
        moment cannot run fixed whatever the count."""
        if self.nelec is None or (self.polarized and self.mode == "noncollinear"):
            return False
        if not self.polarized:
            return True
        if self.requested is None:
            return self.compensated
        return self.requested == int(self.requested)


def _spin(
    magnetic: FieldState[MagneticConfigFacts] | None,
    electron_count: FieldState[ElectronCountDecision] | None,
) -> _Spin:
    """The spin facts of ``magnetic`` and the count of ``electron_count``,
    each read only when resolved."""
    nelec = (
        electron_count.value.nelec
        if electron_count is not None and electron_count.ok
        else None
    )
    if magnetic is None or not magnetic.ok:
        return _Spin(
            mode="none",
            polarized=False,
            requested=None,
            compensated=False,
            nelec=nelec,
            from_magnetic=False,
        )
    facts = magnetic.value
    return _Spin(
        mode=facts.spin_mode,
        polarized=facts.spin_polarized,
        requested=facts.tot_magnetization,
        compensated=facts.compensated,
        nelec=nelec,
        from_magnetic=True,
    )


class OccupationsHumanInput(HumanInput):
    """``smearing_type``/``degauss`` (#35, v2 epic 9, #9) are validated
    here, at construction, not deep inside this module's own logic --
    the same ``set_overrides.build_overrides`` boundary that already
    turns a bad ``occupations`` enum member into a clean
    ``InvalidSetting``. ``smearing_type`` takes ``types.SmearingType``,
    QE's four smearing names, so a name pw.x stops on (#200) is refused
    here rather than rendered; ``degauss`` must be a real, positive
    smearing width."""

    occupations: Occupations | None = None
    smearing_type: SmearingType | None = None
    degauss: float | None = Field(default=None, gt=0)


class OccupationsLlmInput(LlmInput):
    occupations: Occupations | None = None


@dataclass(frozen=True, slots=True)
class OccupationsProtocol:
    """A task protocol's occupations for one of its steps (#174), such as
    the dos task's ``tetrahedra_opt`` nscf step: ranked below a request's
    ``occupations`` and above the llm and the ``is_metal`` rule. Not a
    request value, so its record reads ``heuristic``, with ``note`` naming
    the protocol as its ``source_note``."""

    occupations: Occupations
    note: str


def occupations(
    is_metal: FieldState[Metallicity],
    magnetic: FieldState[MagneticConfigFacts] | None = None,
    electron_count: FieldState[ElectronCountDecision] | None = None,
    human: OccupationsHumanInput | None = None,
    llm: OccupationsLlmInput | None = None,
    protocol: OccupationsProtocol | None = None,
) -> FieldState[OccupationsDecision]:
    """The occupations scheme and the moment it pins (module docstring):
    a human choice, else the step's ``protocol``, else an llm choice, else
    smearing for a metal or an unconfirmed one, else, for a non-metal, fixed
    where fixed occupations can run and smearing otherwise. ``Blocked`` only
    when ``is_metal`` is and nothing chose. The provenance names who set
    each applied setting and the records the decision followed: for the
    heuristic choice, ``is_metal`` and, for a non-metal, the records that
    decided whether fixed occupations can run (``_Spin.fixed_runs_records``);
    on every branch, the records a pinned moment came from (``_provenance``,
    #173). A protocol's choice reads ``heuristic`` with the protocol's note
    (#174)."""
    human = human or OccupationsHumanInput()
    llm = llm or OccupationsLlmInput()
    spin = _spin(magnetic, electron_count)

    if human.occupations is not None:
        decision = _decision_for(
            human.occupations, human.smearing_type, human.degauss, spin
        )
        return Resolved(decision, _provenance(decision, human, None, spin))
    if protocol is not None:
        decision = _decision_for(
            protocol.occupations, human.smearing_type, human.degauss, spin
        )
        return Resolved(
            decision,
            _provenance(decision, human, None, spin, source_note=protocol.note),
        )
    if llm.occupations is not None:
        # This branch reads no human smearing value.
        decision = _decision_for(llm.occupations, None, None, spin)
        return Resolved(decision, _provenance(decision, None, llm, spin))

    if isinstance(is_metal, Blocked):
        return Blocked(by=is_metal)
    non_metal = is_metal.ok and is_metal.value == "non_metal"
    if non_metal and spin.fixed_runs:
        decision = _fixed_decision(spin)
    else:
        # A metal, an unconfirmed one (Unavailable: "heuristic tried, could
        # not tell") and a non-metal whose fixed occupations could not run
        # all land here: smearing is the safe universal default, since
        # unlike fixed it depends neither on the classification being right
        # nor on a known moment (#138). Treating "could not tell" as "assume
        # non-metal" would risk exactly the silent-wrong-answer failure mode
        # fixed occupations produces on an actual metal -- see this module's
        # docstring.
        decision = _smearing_decision(human.smearing_type, human.degauss, spin)
    rule = ("is_metal", *spin.fixed_runs_records) if non_metal else ("is_metal",)
    return Resolved(decision, _provenance(decision, human, None, spin, rule))


def _provenance(
    decision: OccupationsDecision,
    human: OccupationsHumanInput | None,
    llm: OccupationsLlmInput | None,
    spin: _Spin,
    rule: tuple[str, ...] = (),
    source_note: str | None = None,
) -> Provenance:
    """``decision``'s provenance from the inputs its branch read (#173):
    ``occupations`` always, ``smearing_type`` and ``degauss`` only under
    smearing, each ``human`` or ``llm`` where that input set it, else
    ``heuristic``. ``derived_from`` names the records the decision followed:
    ``rule``, those the heuristic choice read, then ``magnetic`` where
    ``decision`` pins a moment -- the magnetic record's admitted request, or
    the 0 of its compensated ordering -- and ``electron_count`` where that 0
    needed an even count. ``source_note`` names the protocol that chose
    (#174)."""
    applied = (
        ("occupations", "smearing_type", "degauss")
        if decision.occupations == "smearing"
        else ("occupations",)
    )
    derived = list(rule)
    if decision.tot_magnetization is not None:
        derived.append("magnetic")
        if spin.requested is None:
            derived.append("electron_count")
    return Provenance.from_field_sources(
        field_sources_from(human, llm, fields=applied),
        derived_from=tuple(dict.fromkeys(derived)),
        source_note=source_note,
    )


def _decision_for(
    choice: Occupations,
    smearing_type: SmearingType | None,
    degauss: float | None,
    spin: _Spin,
) -> OccupationsDecision:
    if choice == "fixed":
        return _fixed_decision(spin)
    if choice == "smearing":
        return _smearing_decision(smearing_type, degauss, spin)
    return OccupationsDecision(
        occupations="tetrahedra_opt",
        smearing_type=None,
        degauss=None,
        tot_magnetization=spin.requested,
    )


def _fixed_decision(spin: _Spin) -> OccupationsDecision:
    """``fixed`` with the moment it pins, warning where a collinear run
    has none for the gate's integer-moment rule to check (a noncollinear
    run takes none, #146)."""
    moment = spin.fixed_moment
    warnings: tuple[Warning, ...] = ()
    if spin.mode == "collinear" and moment is None:
        warnings = (FIXED_SPIN_POLARIZED_WARNING,)
    return OccupationsDecision(
        occupations="fixed",
        smearing_type=None,
        degauss=None,
        tot_magnetization=moment,
        warnings=warnings,
    )


def _smearing_decision(
    smearing_type: SmearingType | None, degauss: float | None, spin: _Spin
) -> OccupationsDecision:
    """``smearing_type``/``degauss`` (v2 epic 9, #9, #34): honored even
    when ``occupations`` itself lands on ``smearing`` through the
    heuristic default, not only when a caller also sets
    ``occupations="smearing"`` explicitly -- a human
    ``smearing_type``/``degauss`` override with no matching
    ``occupations`` override used to be silently discarded in favor of
    the hardcoded 'cold'/0.01 defaults. Carries the human moment only
    (module docstring)."""
    return OccupationsDecision(
        occupations="smearing",
        smearing_type=smearing_type or "cold",
        degauss=degauss if degauss is not None else _METALLIC_DEGAUSS,
        tot_magnetization=spin.requested,
    )
