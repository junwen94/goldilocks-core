"""k_sampling: Gamma-centered k-point mesh, from is_metal (heuristic tier).

New in v2 (v2 epic 6, #6). Second of five per-step advisors
(``occupations -> k_sampling -> n_irr_k -> nbnd -> convergence``,
goldilocks-core-design.md:3239-3248) -- named ``k_sampling`` rather than
v1's ambiguous ``sampling``/``k_spacing`` (that name is reserved for a
future ``q_sampling.py`` covering phonon q-points and hybrid-functional
EXX q-grids, which are a genuinely different thing).

v1 had **no heuristic tier at all** for k-point selection: both of its
advisors (``advice/kdistance.py``'s QRF backend, ``advice/kindex.py``)
are pure ML, with any model/feature failure propagating as a raw
exception straight out of ``resolve_kpoints`` (e.g.
``ml/qrf/inference.py`` raises bare ``ValueError``s on malformed
quantiles, with nothing catching them between there and the caller).
This module's job is to guarantee a working, zero-ML-dependency path
always exists underneath whatever the ml tier does.

The ml tier itself (v2 epic 11, #11; #92) calls QRF95 -- registered in
``ml/registry.toml``'s ``[defaults.kpoints]`` since before this epic,
but never actually loaded until now: its own PSDI record always carried
a real ``goldilocks_ml.inference``-shaped ``model.json``, it just was
not one of this asset's registered files. ``ml.predict.predict_k_distance``
degrades to ``None`` (falling through to heuristic) the same way
``is_metal``/``is_magnetic`` do for a missing asset or import -- see
that module's docstring. A *ladder-rung* k_index model also exists in
goldilocks-ml but is not yet published/wired (#90); that is a distinct
model from QRF95, not another name for it, despite ``capabilities.py``
having (incorrectly) pointed ``k_distance``'s ``ml_target`` at
``"k_index"`` until this same change fixed it.

``occupations`` is accepted as an explicit input -- not because either
the heuristic or the now-wired QRF95 ml tier uses it (neither does;
QRF95's own feature contract is composition/structure/SOAP/lattice/
metallicity, with no smearing term, and the heuristic below is a flat
k_distance regardless of smearing width) -- but because the point of
this epic's per-step ordering is to make sigma an *explicit* input to
k_sampling rather than an implicit shared condition (the goldilocks
dataset itself was generated at fixed ``cold``/0.01 Ry smearing while
only ``k_index`` was scanned, so pretending the two are decoupled would
be dishonest). A future k_index ladder-rung model's own contract (#90)
may turn out to actually consume this; QRF95 does not.

Reuses ``legacy_kmesh/resolve.py``'s human-hint-wins-over-model
precedence pattern (explicit grid beats an explicit distance, both beat
the model/heuristic) as a design template, even though the ladder math
underneath it is entirely new (``kmesh.py``, this epic's other
deliverable).

**Heuristic default k_distance** (operator-specified 2026-09-13, not
from a published benchmark): metals get ``0.15`` A^-1; everything else
-- a *confirmed* non-metal, or ``is_metal`` itself ``Unavailable`` --
gets ``0.30`` A^-1. Unlike ``advisors/occupations.py``, uncertain
metallicity here leans toward the *metal* value, not the cheaper one:
using too coarse a mesh on an actual, undetected metal risks a silently
under-converged answer, while using too fine a mesh on an actual
insulator only costs more compute -- a wrong-but-plausible-looking
number is worse than an expensive-but-correct one. ``fixed`` occupations
misjudging a metal fails differently (typically loud non-convergence),
which is why that module could afford to lean the other way; the two
are not required to agree with each other's tie-break, they just each
need their own honest justification.

Not attempted here (and not claimed to be): the ``is_metal``-based
k_distance *ceiling* and the rung-cannot-exceed-the-ladder clamp
(goldilocks-core-design.md:1732-1817) both gate the future ML tier's
output, not this heuristic's; and dimensionality-aware anisotropic
sampling (e.g. a single k-point along a 2d structure's vacuum direction)
would need to identify *which* lattice vector is non-periodic, which
``analysis/geometry.py`` does not currently expose -- this heuristic is
isotropic and does not special-case low-dimensional structures.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Annotated

from pydantic import Field
from pymatgen.core import Structure

from goldilocks_core.advisors.occupations import OccupationsDecision
from goldilocks_core.analysis.is_metal import Metallicity
from goldilocks_core.inputs.overrides import HumanInput, LlmInput
from goldilocks_core.kmesh import (
    MIN_K_DISTANCE,
    build_gamma_kmesh_entries,
    k_distance_to_mesh,
)
from goldilocks_core.resolution import (
    Blocked,
    FieldState,
    Provenance,
    Resolved,
    Warning,
)

_PositiveInt = Annotated[int, Field(gt=0)]
_ZeroOrOne = Annotated[int, Field(ge=0, le=1)]
"""Not `Literal[0, 1]` (`types.KPointShift`): `capabilities.py`'s
`_json_type` renders any `Literal` as a string enum, which would be
wrong for an integer domain -- a `ge`/`le`-bounded int keeps the
exposed schema's `items` type as `integer`."""

_GAMMA_SHIFT = (0, 0, 0)
_METAL_K_DISTANCE = 0.15
_NON_METAL_K_DISTANCE = 0.30

GRID_AND_DISTANCE_WARNING = Warning(
    code="k_sampling.grid_and_distance_conflict",
    level="info",
    category="k_sampling",
    message=(
        "more than one k-sampling convention was given; k_grid wins "
        "outright and the others are ignored."
    ),
)

INDEX_AND_DISTANCE_WARNING = Warning(
    code="k_sampling.index_and_distance_conflict",
    level="info",
    category="k_sampling",
    message=(
        "both k_index and k_distance were given; k_index wins outright "
        "and k_distance is ignored."
    ),
)

WARNING_CATALOGUE = (GRID_AND_DISTANCE_WARNING, INDEX_AND_DISTANCE_WARNING)
"""Every warning code this module can emit -- ``capabilities.py``'s
``warnings[]`` catalogue aggregates one of these tuples per advisor."""


@dataclass(frozen=True, slots=True)
class KSamplingDecision:
    mesh: tuple[int, int, int]
    shift: tuple[int, int, int]
    k_distance: float | None
    warnings: tuple[Warning, ...] = ()
    k_index: int | None = None
    """Best-effort reverse lookup of the resolved ``mesh`` onto this
    structure's own k-mesh ladder (``kmesh.build_gamma_kmesh_entries``) --
    see ``_with_k_index`` below. ``None`` when the resolved mesh doesn't
    land exactly on a rung (expected whenever the mesh came from
    ``k_distance``/ml/heuristic rather than an explicit ``k_index``
    override), not a failure."""
    k_distance_interval: tuple[float, float] | None = None
    """The matched rung's own ``KMeshEntry.k_distance_interval``, carried
    alongside ``k_index`` whenever a rung is known. Exists so a caller
    resolving via ``k_index`` -- where ``k_distance`` itself is
    deliberately ``None``, a whole interval maps to one rung, not one
    point -- still has something honest to show instead of nothing at
    all; showing a fabricated single point (e.g. the interval midpoint)
    would be worse than showing no number."""


class KSamplingHumanInput(HumanInput):
    k_grid: tuple[_PositiveInt, _PositiveInt, _PositiveInt] | None = None
    """Each axis count must be >= 1 (#-- found in the epic 5/6/7 delivery
    -layer audit): a 0 or negative entry used to be accepted, resolved,
    and written verbatim into the K_POINTS card, silently telling QE to
    sample zero or a negative number of points along that axis."""
    k_index: _PositiveInt | None = None
    """A specific 1-based rung on this structure's own k-mesh ladder
    (``kmesh.build_gamma_kmesh_entries`` -- rung 1 is the Gamma-only
    mesh), for a human who thinks in ladder position rather than a raw
    target spacing. A third, equally explicit convention alongside
    ``k_grid``/``k_distance``, not a variant of either."""
    k_distance: float | None = Field(default=None, gt=0)
    shift: tuple[_ZeroOrOne, _ZeroOrOne, _ZeroOrOne] | None = None
    """QE's own K_POINTS automatic card requires each shift component to
    be exactly 0 or 1 (INPUT_PW.txt, checked 2026-09-14) -- any other
    integer used to be accepted and written verbatim."""


class KSamplingLlmInput(LlmInput):
    k_distance: float | None = Field(default=None, gt=0)


def k_sampling(
    is_metal: FieldState[Metallicity],
    structure: Structure,
    occupations: FieldState[OccupationsDecision] | None = None,
    human: KSamplingHumanInput | None = None,
    llm: KSamplingLlmInput | None = None,
) -> FieldState[KSamplingDecision]:
    human = human or KSamplingHumanInput()
    llm = llm or KSamplingLlmInput()

    if human.k_grid is not None:
        warnings = (
            (GRID_AND_DISTANCE_WARNING,)
            if human.k_index is not None or human.k_distance is not None
            else ()
        )
        decision = KSamplingDecision(
            mesh=human.k_grid,
            shift=human.shift or _GAMMA_SHIFT,
            k_distance=None,
            warnings=warnings,
        )
        return Resolved(_with_k_index(decision, structure), Provenance(source="human"))
    if human.k_index is not None:
        resolved = _from_k_index(structure, human.k_index, human.shift)
        if isinstance(resolved, str):
            return Blocked(by=resolved)
        warnings = (INDEX_AND_DISTANCE_WARNING,) if human.k_distance is not None else ()
        decision = KSamplingDecision(
            mesh=resolved.mesh,
            shift=resolved.shift,
            k_distance=resolved.k_distance,
            warnings=warnings,
            # Already known exactly -- this *is* the rung the human asked
            # for, no reverse lookup needed (and it would be redundant:
            # building the ladder a second time to find what we already
            # have). `_from_k_index` already attached both from the same
            # matched `KMeshEntry`.
            k_index=resolved.k_index,
            k_distance_interval=resolved.k_distance_interval,
        )
        return Resolved(decision, Provenance(source="human"))
    if human.k_distance is not None:
        return Resolved(
            _with_k_index(
                _from_k_distance(structure, human.k_distance, human.shift), structure
            ),
            Provenance(source="human"),
        )
    ml_value = _ml_k_distance(structure)
    if ml_value is not None:
        return Resolved(
            _with_k_index(
                _from_k_distance(structure, ml_value, human.shift), structure
            ),
            Provenance(source="ml"),
        )
    if llm.k_distance is not None:
        return Resolved(
            _with_k_index(_from_k_distance(structure, llm.k_distance, None), structure),
            Provenance(source="llm"),
        )

    if isinstance(is_metal, Blocked):
        return Blocked(by=is_metal)
    if is_metal.ok and is_metal.value == "non_metal":
        return Resolved(
            _with_k_index(
                _from_k_distance(structure, _NON_METAL_K_DISTANCE, human.shift),
                structure,
            ),
            Provenance(source="heuristic"),
        )
    return Resolved(
        _with_k_index(
            _from_k_distance(structure, _METAL_K_DISTANCE, human.shift), structure
        ),
        Provenance(source="heuristic"),
    )


def _ml_k_distance(structure: Structure) -> float | None:
    """QRF95's own raw k-distance (v2 epic 11, #11; #92), or ``None`` if
    its model asset is not installed or goldilocks-ml is not importable
    -- never a reason to fail ``k_sampling()`` itself, the same
    degrade-to-heuristic policy ``analysis/is_metal.py``'s
    ``_ml_is_metal`` already follows for a missing external dependency."""
    from goldilocks_core.ml.predict import MlModelUnavailable, predict_k_distance

    try:
        prediction = predict_k_distance(structure)
    except MlModelUnavailable:
        return None
    return float(prediction.value)


def _from_k_distance(
    structure: Structure, k_distance: float, shift: tuple[int, int, int] | None
) -> KSamplingDecision:
    return KSamplingDecision(
        mesh=k_distance_to_mesh(structure, k_distance),
        shift=shift or _GAMMA_SHIFT,
        k_distance=k_distance,
    )


def _from_k_index(
    structure: Structure, k_index: int, shift: tuple[int, int, int] | None
) -> KSamplingDecision | str:
    """Resolve an explicit 1-based ladder rung to its concrete mesh via
    ``kmesh.build_gamma_kmesh_entries``, or return why not (a rung
    beyond this structure's own ladder length -- the ladder terminates
    at ``MIN_K_DISTANCE``, so how many rungs exist is structure
    -dependent, not a fixed ceiling a human could know in advance).
    ``k_distance`` on the returned decision is ``None``, the same as
    ``k_grid``'s: the mesh was picked by rung, not by a target spacing,
    so there is no single distance value to report."""
    entries = build_gamma_kmesh_entries(structure)
    if k_index > len(entries):
        return (
            f"k_index={k_index} exceeds this structure's own ladder length "
            f"({len(entries)} rungs down to the {MIN_K_DISTANCE} A^-1 floor)"
        )
    entry = entries[k_index - 1]
    return KSamplingDecision(
        mesh=entry.mesh,
        shift=shift or _GAMMA_SHIFT,
        k_distance=None,
        k_index=entry.kindex,
        k_distance_interval=entry.k_distance_interval,
    )


def _with_k_index(
    decision: KSamplingDecision, structure: Structure
) -> KSamplingDecision:
    """Best-effort reverse lookup: does the resolved ``mesh`` land exactly
    on a rung of this structure's own ladder? ``build_gamma_kmesh_entries``
    guarantees the ladder is complete and non-repeating (its own
    docstring), so at most one entry can match and the first hit is
    authoritative. Leaves ``decision`` untouched (``k_index`` stays
    ``None``) when the mesh came from a ``k_distance``/ml/heuristic value
    that falls between rungs -- not every resolved mesh has one."""
    for entry in build_gamma_kmesh_entries(structure):
        if entry.mesh == decision.mesh:
            return replace(
                decision,
                k_index=entry.kindex,
                k_distance_interval=entry.k_distance_interval,
            )
    return decision
