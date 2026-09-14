"""hubbard_u: whether a Hubbard +U correction is needed, and how to get the
U value, from needs_correlation and composition.

New in v2 (v2 epic 5, #5). Two non-ml fallback plans
(goldilocks-core-design.md:2610-2613), both represented as `Resolved` outcomes
-- an advisor successfully deciding "use this" or "here is what a human/agent
needs to do next" is not the same as `Unavailable`, which means the heuristic
tried and could not tell anything at all:

- **Plan A** -- a small package-default table of commonly-published
  Dudarev `U_eff` values for 3d transition-metal oxides (Wang et al. 2006 /
  Materials Project's own long-standing settings). Used when every
  correlated element in the structure is in this table.
- **Plan B** -- when an element is not in the table (4d/5d transition
  metals, lanthanides, actinides, or any 3d element this package does not
  carry a value for), this produces a `CalibrationRequest` per element
  instead of guessing: which element, which orbital manifold, and enough
  detail for the QE-generation layer to eventually render an `hp.x`
  self-consistent-linear-response input and a README explaining how to run
  it (runnable via `aiida-hubbard` if the calling agent has AiiDA capability).
  Rendering that input file is `generation/`'s job (v2 epic 7's "QE
  generation rewrite"), not this advisor's -- `advisors/` decide settings,
  they do not write files. This is why `CalibrationRequest` carries
  structured fields rather than a free-text note: epic 7 needs to consume
  it, not just display it.

**No `ml` tier, permanently, not "not yet built."** An ENN-based +U predictor
(Uhrin et al. 2025) was evaluated and rejected in the design doc
(:2635): it needs an SCF-derived occupation matrix as input, and every
`advisors/` function in this codebase is a stateless per-call function that
never has one to give it.

**Cross-parameter rule** (:2626-2627): a hybrid functional already handles
the self-interaction error +U exists to patch, so this returns "not needed"
whenever `functional` is `advisors.functional.HYBRID_FUNCTIONAL`, regardless
of what `needs_correlation` says.

Also implements `expand_hubbard_label`
(goldilocks-core-design.md:2673-2684, closing
goldilocks-qe-pw-parameter-audit.md's P0 finding #1): a naive `U Fe-3d 4.6`
HUBBARD card line is not valid QE input once AFM species-splitting produces
`Fe1`/`Fe2` in `ATOMIC_SPECIES` -- QE requires the HUBBARD label to match
`ATOMIC_SPECIES` verbatim. Same-element split species default to the same U
value unless overridden. `magnetic_config.py`'s `relabeled_structure` ships
FM-only (identity) for now, so this has no real split species to expand yet
in this epic -- exercised directly with a manually-split structure in its
own tests, the same pattern `analysis/needs_soc.py` used for a dependency
that also has no real failure to trigger it yet.

**Note for epic 7 (QE generation rewrite), re: the aiida-quantumespresso
hubbard.html tutorial.** QE >= 7.1's actual `HUBBARD` card is not the
label-keyed `U Fe-3d 4.6` style `expand_hubbard_label` targets above -- it is
atom-*index*-keyed: ``HUBBARD ortho-atomic`` followed by lines like
``V Co-3d Co-3d 1 1 5.0`` (``V <manifold_I> <manifold_J> <site_i> <site_j>
<value>``; onsite U is the degenerate case ``site_i == site_j``). This is
also the format `hp.x`/`aiida-hubbard` itself emits from a calibration run.
If epic 7 targets this format, AFM-split *species labels* stop being the
mechanism for giving symmetry-inequivalent sites of the same element
different U values -- distinct site indices already do that, with no
`ATOMIC_SPECIES` relabeling required. Whether to target this format or the
older label-keyed one is epic 7's decision to make once it exists;
`expand_hubbard_label` here is not wrong for what it claims to do (expand a
label-keyed table), it just may not be the table shape the renderer ends up
needing. Separately, `hp.x` also requires its "Hubbard atoms" to be listed
first in `ATOMIC_POSITIONS` -- a structure-preprocessing detail for whatever
in epic 7 renders the calibration input, not a concern of this advisor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element

from goldilocks_core.advisors.functional import HYBRID_FUNCTIONAL
from goldilocks_core.analysis.composition import CompositionFacts, composition
from goldilocks_core.inputs.overrides import HumanInput, LlmInput
from goldilocks_core.resolution import (
    Blocked,
    FieldState,
    Provenance,
    Resolved,
    Warning,
    blocked_by,
)

HubbardPlan = Literal["not_needed", "table", "self_consistent_calibration_needed"]

_MP_DUDAREV_U_EFF = {
    "Co": 3.32,
    "Cr": 3.7,
    "Cu": 4.0,
    "Fe": 5.3,
    "Mn": 3.9,
    "Ni": 6.2,
    "V": 3.25,
}
"""Commonly-published Dudarev U_eff (eV) for 3d transition-metal oxides
(Wang, Maxisch & Ceder, Phys. Rev. B 73, 195107 (2006); Materials Project's
own long-standing GGA+U settings). A package default for common cases, not
an exhaustive or authoritative table -- deliberately narrow rather than
guessed-at for elements this package has no citation for."""

_MANIFOLD_BY_BLOCK_AND_ROW = {
    ("d", 4): "3d",
    ("d", 5): "4d",
    ("d", 6): "5d",
    ("f", 6): "4f",
    ("f", 7): "5f",
}
"""(pymatgen Element.block, Element.row) -> Hubbard manifold label. Row 4
d-block is 3d (Sc-Zn); row 6/7 f-block is lanthanides/actinides (pymatgen
classes La itself as both a lanthanide and a transition metal -- its row/
block still resolve to 4f here, which is what a Hubbard correction on La
would actually target)."""

PROJECTOR_MISMATCH_WARNING = Warning(
    code="hubbard.projector_mismatch",
    level="warning",
    category="hubbard",
    message=(
        "Materials Project's U values are calibrated for VASP's PAW projectors; "
        "QE's atomic/ortho-atomic/norm-atomic projectors are not guaranteed to "
        "give the same effective U for the same physical correction."
    ),
)

STARTING_NS_EIGENVALUE_WARNING = Warning(
    code="hubbard.starting_ns_eigenvalue_missing",
    level="warning",
    category="hubbard",
    message=(
        "No starting_ns_eigenvalue initial occupation-matrix guess is provided "
        "(pitfall A10): +U calculations have multiple metastable occupation-matrix "
        "solutions and may converge to the wrong one silently. Supplying an "
        "explicit initial guess is recommended but not computed automatically here."
    ),
)
"""A10 (goldilocks-core-design.md:4465): a real initial guess needs a
crystal-field/atomic-physics decomposition of the occupation matrix by spin
channel this package does not attempt -- recorded as an explicit warning
(the design doc's own instruction to record this into provenance) rather
than a guessed number, which is the honest answer tri-state exists to
allow."""

WARNING_CATALOGUE = (PROJECTOR_MISMATCH_WARNING, STARTING_NS_EIGENVALUE_WARNING)
"""Every warning code this module can emit -- ``capabilities.py``'s
``warnings[]`` catalogue aggregates one of these tuples per advisor."""


@dataclass(frozen=True, slots=True)
class CalibrationRequest:
    element: str
    manifold: str
    suggested_method: Literal["hp.x", "aiida-hubbard"] = "hp.x"
    note: str = (
        "No package-default U value for this element/manifold. Self-consistent "
        "linear-response calibration (hp.x, or aiida-hubbard if the agent has "
        "AiiDA capability) is recommended before publishing a +U calculation."
    )


@dataclass(frozen=True, slots=True)
class HubbardUDecision:
    plan: HubbardPlan
    u_by_element: dict[str, float]
    calibration_requests: tuple[CalibrationRequest, ...] = ()
    warnings: tuple[Warning, ...] = ()


class HubbardUHumanInput(HumanInput):
    needs_correlation: bool | None = None
    u_by_element: dict[str, float] | None = None


class HubbardULlmInput(LlmInput):
    needs_correlation: bool | None = None


_NOT_NEEDED = HubbardUDecision(plan="not_needed", u_by_element={})


def hubbard_u(
    structure: Structure,
    needs_correlation: FieldState[bool],
    functional: str,
    human: HubbardUHumanInput | None = None,
    llm: HubbardULlmInput | None = None,
) -> FieldState[HubbardUDecision]:
    human = human or HubbardUHumanInput()
    llm = llm or HubbardULlmInput()

    if functional == HYBRID_FUNCTIONAL:
        return Resolved(_NOT_NEEDED, Provenance(source="heuristic"))

    if human.needs_correlation is False:
        return Resolved(_NOT_NEEDED, Provenance(source="human"))
    if human.u_by_element is not None:
        return Resolved(
            HubbardUDecision(plan="table", u_by_element=dict(human.u_by_element)),
            Provenance(source="human"),
        )
    if human.needs_correlation is True:
        return _heuristic(structure, source="human")
    if llm.needs_correlation is False:
        return Resolved(_NOT_NEEDED, Provenance(source="llm"))
    if llm.needs_correlation is True:
        return _heuristic(structure, source="llm")

    if not needs_correlation.ok:
        return Blocked(by=blocked_by(needs_correlation))
    if not needs_correlation.value:
        return Resolved(_NOT_NEEDED, Provenance(source="heuristic"))
    return _heuristic(structure, source="heuristic")


def _heuristic(structure: Structure, *, source: str) -> FieldState[HubbardUDecision]:
    facts = composition(structure).value
    correlated_elements = _correlated_elements(facts)
    if not correlated_elements:
        # #36 (v2 epic 9, #9): forcing needs_correlation onward with no
        # correlated element in the structure at all (e.g. plain Si)
        # used to fall through to plan="table" with an EMPTY
        # u_by_element plus two Hubbard-specific warnings that make no
        # sense with no +U term present -- and since write_qe_scf
        # treats any plan != "not_needed" as a real +U resolution, this
        # then failed with "a Hubbard +U correction was resolved",
        # which was factually wrong (nothing was). Matches the other
        # not_needed short-circuits in hubbard_u() above.
        return Resolved(_NOT_NEEDED, Provenance(source=source))

    u_by_element: dict[str, float] = {}
    calibration_requests: list[CalibrationRequest] = []
    for symbol in correlated_elements:
        value = _MP_DUDAREV_U_EFF.get(symbol)
        if value is not None:
            u_by_element[symbol] = value
            continue
        manifold = _manifold_for(symbol)
        calibration_requests.append(
            CalibrationRequest(element=symbol, manifold=manifold)
        )

    if calibration_requests:
        return Resolved(
            HubbardUDecision(
                plan="self_consistent_calibration_needed",
                u_by_element=u_by_element,
                calibration_requests=tuple(calibration_requests),
                warnings=(STARTING_NS_EIGENVALUE_WARNING,) if u_by_element else (),
            ),
            Provenance(source=source),
        )
    return Resolved(
        HubbardUDecision(
            plan="table",
            u_by_element=u_by_element,
            warnings=(PROJECTOR_MISMATCH_WARNING, STARTING_NS_EIGENVALUE_WARNING),
        ),
        Provenance(source=source),
    )


def _correlated_elements(facts: CompositionFacts) -> tuple[str, ...]:
    return tuple(
        sorted({*facts.transition_metals, *facts.lanthanides, *facts.actinides})
    )


def _manifold_for(symbol: str) -> str:
    element = Element(symbol)
    manifold = _MANIFOLD_BY_BLOCK_AND_ROW.get((element.block, element.row))
    return manifold or f"{element.row}{element.block}"


def expand_hubbard_label(
    u_by_element: dict[str, float], relabeled_structure: Structure
) -> dict[str, float]:
    """Expand a canonical-element-keyed U table to match
    ``relabeled_structure``'s actual species labels (e.g. ``Fe`` -> ``Fe1``,
    ``Fe2`` once AFM splitting produces them), so a HUBBARD card can name a
    label that genuinely appears in ``ATOMIC_SPECIES``
    (goldilocks-qe-pw-parameter-audit.md P0 finding #1). Split species default
    to their canonical element's U value unless the caller already overrode
    them individually in ``u_by_element``.
    """
    labels = {site.label for site in relabeled_structure}
    expanded = dict(u_by_element)
    for label in labels:
        if label in expanded:
            continue
        canonical = _canonical_element(label)
        if canonical in u_by_element:
            expanded[label] = u_by_element[canonical]
    return expanded


def _canonical_element(label: str) -> str:
    """ "Fe1" -> "Fe": strip a trailing AFM-split-species digit suffix."""
    return label.rstrip("0123456789")
