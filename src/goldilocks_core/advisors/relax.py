"""relax: ion/cell relaxation parameters, for relax/vc-relax tasks.

``RelaxOptions``/``VcRelaxOptions`` (the typed fields) landed in v2 epic
7 (#7); this epic (v2 epic 10, #10) adds the actual decision cascade
(``relax_settings``, human/llm/heuristic precedence like every other
advisor) plus the two ``HumanInput``/``LlmInput`` bags, closing the gap
epic 7's own docstring named ("full relax parameter coverage... is v2
epic 10's job"). Every default is copied verbatim from QE's own
``INPUT_PW.txt`` (re-downloaded and checked 2026-09-14, not carried over
from memory or the design doc's own prose), not invented or "improved"
with unvalidated numbers.

**Not in this epic's scope, on purpose:** ``if_pos`` (fixing bottom
layers of a slab) needs layer detection this codebase does not have yet
-- split out to its own follow-up issue (#44) once the rest of this
epic's scope turned out large enough to implement on its own; the
heuristic default (this module makes no decision at all: ``if_pos``
does not exist as a field yet) still matches QE's own "don't write the
card" default, so nothing here is a regression relative to not shipping
it.

**``etot_conv_thr`` is not this module's own decision.** ``advisors/
convergence.py`` already resolves it as ``nat * etot_conv_thr_per_atom``
(the exact formula/constant the design doc wants for relax too, since
this is the same "how much can the total energy still wobble" question
scf's own convergence advisor already answers) -- ``relax_settings``
reads ``ConvergenceDecision.etot_conv_thr`` off the step's already
-resolved convergence decision rather than recomputing it, so there is
exactly one nat-scaling formula in this codebase, not two that could
drift apart. There is deliberately no separate ``--set``-able
``relax``-scoped ``etot_conv_thr``: the existing ``etot_conv_thr`` key
(``ConvergenceHumanInput``) already reaches this value.

**A real gap found while implementing this epic, not in the original
issue:** ``VcRelaxOptions.cell_dynamics`` (epic 7) is a derived property
that always equals ``ion_dynamics`` -- correct only because this
codebase, until now, could never construct an ``ion_dynamics`` other
than the default ``'bfgs'``. Now that ``RelaxHumanInput``/
``RelaxLlmInput`` exist, a caller *can* set ``ion_dynamics='damp'`` on a
vc-relax request, which would make the derived property emit
``cell_dynamics='damp'`` -- not a valid QE ``cell_dynamics`` value for
vc-relax at all (valid values, confirmed against ``INPUT_PW.txt``:
``none``/``sd``/``damp-pr``/``damp-w``/``bfgs``). ``checks.py``'s
``_vc_relax_requires_bfgs_ion_dynamics`` blocks this rather than letting
it reach generation -- this codebase only models the bfgs/bfgs
combination for vc-relax (the design doc's own stated v2 scope: damp
-paired cell dynamics needs ``wmass``, audit finding F41, "low priority
deferred").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from goldilocks_core.advisors.convergence import ConvergenceDecision
from goldilocks_core.analysis.geometry import GeometryFacts
from goldilocks_core.analysis.symmetry import SymmetryFacts
from goldilocks_core.inputs.overrides import HumanInput, LlmInput
from goldilocks_core.resolution import (
    Blocked,
    FieldState,
    Provenance,
    Resolved,
    Source,
    Warning,
    blocked_by,
)

IonDynamics = Literal["bfgs", "damp", "fire"]
"""QE's full ``ion_dynamics`` domain across ``relax``/``vc-relax``
(``INPUT_PW.txt``, checked 2026-09-14). ``fire`` is documented for
``relax`` only, not ``vc-relax`` -- not separately encoded here because
``checks.py``'s ``_vc_relax_requires_bfgs_ion_dynamics`` already rejects
every non-``'bfgs'`` value for vc-relax, ``fire`` included."""

CellDofree = Literal[
    "all",
    "ibrav",
    "a",
    "b",
    "c",
    "fixa",
    "fixb",
    "fixc",
    "x",
    "y",
    "z",
    "xy",
    "xz",
    "yz",
    "xyz",
    "shape",
    "volume",
    "2Dxy",
    "2Dshape",
    "epitaxial_ab",
    "epitaxial_ac",
    "epitaxial_bc",
    "ibrav+a",
    "ibrav+b",
    "ibrav+c",
    "ibrav+fixa",
    "ibrav+fixb",
    "ibrav+fixc",
    "ibrav+x",
    "ibrav+y",
    "ibrav+z",
    "ibrav+xy",
    "ibrav+xz",
    "ibrav+yz",
    "ibrav+xyz",
    "ibrav+shape",
    "ibrav+volume",
    "ibrav+2Dxy",
    "ibrav+2Dshape",
    "ibrav+epitaxial_ab",
    "ibrav+epitaxial_ac",
    "ibrav+epitaxial_bc",
]
"""Every value QE's ``cell_dofree`` accepts (``INPUT_PW.txt``, checked
2026-09-14): one of the 20 base options, or ``'ibrav'`` plus any base
option other than ``'all'``/``'ibrav'`` themselves -- official text:
"You can use this option [ibrav] in combination with any other one by
specifying 'ibrav+option'". Encoded as an explicit ``Literal`` (matching
this codebase's existing style, e.g. ``types.SmearingType``) rather than
a runtime membership check, so an invalid value is rejected as a normal
pydantic validation error at the transport boundary (#35's pattern), not
a bespoke error path."""

_FORC_CONV_THR_DEFAULT = 1.0e-3
"""a.u. (Ry/Bohr). QE's own official INPUT_PW default."""
_NSTEP_DEFAULT = 50
"""QE's own default for calculation not in {scf, nscf, bands}."""
_TRUST_RADIUS_MAX_DEFAULT = 0.8
"""Bohr. bfgs only."""
_TRUST_RADIUS_MIN_DEFAULT = 1.0e-3
"""Bohr. bfgs only."""
_TRUST_RADIUS_INI_DEFAULT = 0.5
"""Bohr. bfgs only."""
_PRESS_DEFAULT = 0.0
"""kbar."""
_PRESS_CONV_THR_DEFAULT = 0.5
"""kbar."""
_CELL_FACTOR_DEFAULT = 2.0
"""Must exceed the maximum linear *contraction* the cell will undergo,
not expansion (audit finding #3)."""
_DEFAULT_CELL_DOFREE: CellDofree = "all"
_HEXAGONAL_2D_CELL_DOFREE: CellDofree = "ibrav+2Dxy"
_NON_HEXAGONAL_2D_CELL_DOFREE: CellDofree = "2Dxy"

NON_HEXAGONAL_2D_CELL_DOFREE_WARNING = Warning(
    code="relax.cell_dofree_non_hexagonal_2d",
    level="warning",
    category="relax",
    message=(
        "cell_dofree='2Dxy' is being used on a non-hexagonal 2D structure; "
        "QE's own docs warn the symmetry-preserving 'ibrav+2Dxy' form "
        "'will never converge' for non-hexagonal cells, so the bare '2Dxy' "
        "form is used instead, which can silently break exact cell "
        "symmetry during relaxation."
    ),
)

WARNING_CATALOGUE = (NON_HEXAGONAL_2D_CELL_DOFREE_WARNING,)
"""Every warning code this module can emit -- ``capabilities.py``'s
``warnings[]`` catalogue aggregates one of these tuples per advisor."""


@dataclass(frozen=True, slots=True)
class RelaxOptions:  # calculation ∈ {relax, vc-relax}
    ion_dynamics: IonDynamics = "bfgs"
    forc_conv_thr: float = _FORC_CONV_THR_DEFAULT
    """a.u. (Ry/Bohr). QE's own official INPUT_PW default."""
    etot_conv_thr: float = 1.0e-5
    """a.u. Not this module's own default -- ``relax_settings`` always
    overwrites this with the caller's already-resolved
    ``ConvergenceDecision.etot_conv_thr`` (``nat *
    etot_conv_thr_per_atom``, this module's own docstring). This bare
    default only matters for a ``RelaxOptions`` built by hand (tests),
    never a real ``relax_settings`` output."""
    nstep: int = _NSTEP_DEFAULT
    trust_radius_max: float = _TRUST_RADIUS_MAX_DEFAULT
    """Bohr."""
    trust_radius_min: float = _TRUST_RADIUS_MIN_DEFAULT
    """Bohr."""
    trust_radius_ini: float = _TRUST_RADIUS_INI_DEFAULT
    """Bohr."""
    remove_rigid_rot: bool = False
    warnings: tuple[Warning, ...] = ()


@dataclass(frozen=True, slots=True)
class VcRelaxOptions(RelaxOptions):  # calculation == 'vc-relax'
    cell_dofree: CellDofree = _DEFAULT_CELL_DOFREE
    press: float = _PRESS_DEFAULT
    """kbar."""
    press_conv_thr: float = _PRESS_CONV_THR_DEFAULT
    """kbar."""
    cell_factor: float = _CELL_FACTOR_DEFAULT
    """Must exceed the maximum linear *contraction* the cell will
    undergo, not expansion -- see this module's docstring, audit
    finding #3."""

    @property
    def cell_dynamics(self) -> str:
        """Always equals ``ion_dynamics`` -- QE requires this coupling
        (audit finding B5, confirmed both directions in ``INPUT_PW.txt``
        2026-09-14). A derived property, not an independent field, so
        the invalid combination cannot be constructed. Only ever
        ``'bfgs'`` in practice: ``checks.py`` blocks any other
        ``ion_dynamics`` for vc-relax -- see this module's own
        docstring."""
        return self.ion_dynamics


class RelaxHumanInput(HumanInput):
    ion_dynamics: IonDynamics | None = None
    forc_conv_thr: float | None = Field(default=None, gt=0)
    nstep: int | None = Field(default=None, gt=0)
    trust_radius_max: float | None = Field(default=None, gt=0)
    trust_radius_min: float | None = Field(default=None, gt=0)
    trust_radius_ini: float | None = Field(default=None, gt=0)
    remove_rigid_rot: bool | None = None
    cell_dofree: CellDofree | None = None
    press: float | None = None
    press_conv_thr: float | None = Field(default=None, gt=0)
    cell_factor: float | None = Field(default=None, gt=0)


class RelaxLlmInput(LlmInput):
    """Numerically-sensitive thresholds (``forc_conv_thr``,
    ``trust_radius_*``, ``press_conv_thr``, ``cell_factor``) are
    human-only, matching ``advisors/convergence.py``'s own
    ``ConvergenceLlmInput`` precedent of keeping convergence-threshold
    -shaped fields out of the llm tier -- only "which scenario" choices
    are here."""

    ion_dynamics: IonDynamics | None = None
    nstep: int | None = None
    remove_rigid_rot: bool | None = None
    cell_dofree: CellDofree | None = None
    press: float | None = None


def relax_settings(
    calculation: Literal["relax", "vc-relax"],
    convergence: FieldState[ConvergenceDecision],
    symmetry: FieldState[SymmetryFacts] | None = None,
    geometry: FieldState[GeometryFacts] | None = None,
    human: RelaxHumanInput | None = None,
    llm: RelaxLlmInput | None = None,
) -> FieldState[RelaxOptions | VcRelaxOptions]:
    """``etot_conv_thr`` makes this ``Blocked`` whenever ``convergence``
    itself is not ``Resolved`` -- there is no fallback constant, per
    this module's own docstring on why that value is read off
    ``convergence``, not recomputed."""
    human = human or RelaxHumanInput()
    llm = llm or RelaxLlmInput()

    if not convergence.ok:
        return Blocked(by=blocked_by(convergence))

    convergence_field_sources = convergence.provenance.field_sources or {}
    etot_conv_thr_source = convergence_field_sources.get(
        "etot_conv_thr", convergence.source
    )

    warnings: list[Warning] = []
    base_kwargs: dict[str, object] = {
        "ion_dynamics": _pick(human.ion_dynamics, llm.ion_dynamics, "bfgs"),
        "forc_conv_thr": _pick(human.forc_conv_thr, None, _FORC_CONV_THR_DEFAULT),
        "etot_conv_thr": convergence.value.etot_conv_thr,
        "nstep": _pick(human.nstep, llm.nstep, _NSTEP_DEFAULT),
        "trust_radius_max": _pick(
            human.trust_radius_max, None, _TRUST_RADIUS_MAX_DEFAULT
        ),
        "trust_radius_min": _pick(
            human.trust_radius_min, None, _TRUST_RADIUS_MIN_DEFAULT
        ),
        "trust_radius_ini": _pick(
            human.trust_radius_ini, None, _TRUST_RADIUS_INI_DEFAULT
        ),
        "remove_rigid_rot": _pick(human.remove_rigid_rot, llm.remove_rigid_rot, False),
    }
    field_sources: dict[str, Source] = {
        "ion_dynamics": _field_source(human.ion_dynamics, llm.ion_dynamics),
        "forc_conv_thr": _field_source(human.forc_conv_thr),
        "etot_conv_thr": etot_conv_thr_source,
        "nstep": _field_source(human.nstep, llm.nstep),
        "trust_radius_max": _field_source(human.trust_radius_max),
        "trust_radius_min": _field_source(human.trust_radius_min),
        "trust_radius_ini": _field_source(human.trust_radius_ini),
        "remove_rigid_rot": _field_source(human.remove_rigid_rot, llm.remove_rigid_rot),
    }

    if calculation == "vc-relax":
        cell_dofree, cell_dofree_source, cell_dofree_warning = _resolve_cell_dofree(
            human.cell_dofree, llm.cell_dofree, symmetry, geometry
        )
        if cell_dofree_warning is not None:
            warnings.append(cell_dofree_warning)
        decision: RelaxOptions | VcRelaxOptions = VcRelaxOptions(
            **base_kwargs,
            cell_dofree=cell_dofree,
            press=_pick(human.press, llm.press, _PRESS_DEFAULT),
            press_conv_thr=_pick(human.press_conv_thr, None, _PRESS_CONV_THR_DEFAULT),
            cell_factor=_pick(human.cell_factor, None, _CELL_FACTOR_DEFAULT),
            warnings=tuple(warnings),
        )
        field_sources.update(
            {
                "cell_dofree": cell_dofree_source,
                "press": _field_source(human.press, llm.press),
                "press_conv_thr": _field_source(human.press_conv_thr),
                "cell_factor": _field_source(human.cell_factor),
            }
        )
    else:
        decision = RelaxOptions(**base_kwargs, warnings=tuple(warnings))

    source = "human" if _any_set(human) else "llm" if _any_set(llm) else "heuristic"
    return Resolved(decision, Provenance(source=source, field_sources=field_sources))


def _resolve_cell_dofree(
    human_value: CellDofree | None,
    llm_value: CellDofree | None,
    symmetry: FieldState[SymmetryFacts] | None,
    geometry: FieldState[GeometryFacts] | None,
) -> tuple[CellDofree, Source, Warning | None]:
    if human_value is not None:
        return human_value, "human", None
    if llm_value is not None:
        return llm_value, "llm", None
    is_2d = bool(
        geometry is not None and geometry.ok and geometry.value.dimensionality == "2d"
    )
    if not is_2d:
        return _DEFAULT_CELL_DOFREE, "heuristic", None
    is_hexagonal = bool(
        symmetry is not None
        and symmetry.ok
        and symmetry.value.crystal_system == "hexagonal"
    )
    if is_hexagonal:
        return _HEXAGONAL_2D_CELL_DOFREE, "heuristic", None
    return (
        _NON_HEXAGONAL_2D_CELL_DOFREE,
        "heuristic",
        NON_HEXAGONAL_2D_CELL_DOFREE_WARNING,
    )


def _pick[T](human_value: T | None, llm_value: T | None, default: T) -> T:
    if human_value is not None:
        return human_value
    if llm_value is not None:
        return llm_value
    return default


def _field_source(
    human_value: object | None, llm_value: object | None = None
) -> Source:
    if human_value is not None:
        return "human"
    if llm_value is not None:
        return "llm"
    return "heuristic"


def _any_set(overrides: HumanInput) -> bool:
    return bool(overrides.model_dump(exclude_defaults=True))
