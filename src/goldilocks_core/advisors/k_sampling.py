"""k_sampling: Gamma-centered k-point mesh, from is_metal (heuristic tier).

New in v2 (v2 epic 6, #6). Second of five per-step advisors
(``occupations -> k_sampling -> n_irr_k -> nbnd -> convergence``,
goldilocks-core-design.md:3239-3248) -- named ``k_sampling`` rather than
v1's ambiguous ``sampling``/``k_spacing``: the phonon q-points, a
genuinely different thing, are ``advisors/q_sampling.py``'s.

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
not one of this asset's registered files. When that model is not
installed, ``ml.predict.predict_k_distance`` raises ``MlModelUnavailable``
and this tier falls through to the heuristic, the same way
``is_metal``/``is_magnetic`` do; any other model failure fails the
request (#130) -- see that module's docstring. A *ladder-rung* k_index
model also exists in goldilocks-ml but is not yet published/wired (#90);
it predicts rungs on goldilocks-data's ladder, which core's no longer
matches rung for rung (#163, ``kmesh``), and it is a distinct model from
QRF95, not another name for it, despite
``capabilities.py`` having (incorrectly) pointed ``k_distance``'s
``ml_target`` at ``"k_index"`` until this same change fixed it.

**A step's protocol** (#174). A task can give one of its steps its own
k_distance (``KSamplingProtocol``; the dos task's nscf step,
``service/_step_protocol.py``). It ranks below the request's own
``k_grid``/``k_index``/``k_distance`` and above QRF95 and the llm: QRF95
is trained on scf meshes, and a denser mesh is the point of that step. Its
record reads ``heuristic``, with the protocol named in ``source_note``,
since the request did not set it; a request's ``shift`` still applies on
it.

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
from a published benchmark): metals, and ``is_metal`` itself
``Unavailable``, get ``0.15`` A^-1; a *confirmed* non-metal gets ``0.30``
A^-1. Uncertain metallicity leans toward the *metal* value, not the
cheaper one, as it does in ``advisors/occupations.py``: using too coarse
a mesh on an actual, undetected metal risks a silently under-converged
answer, while using too fine a mesh on an actual insulator only costs
more compute -- a wrong-but-plausible-looking number is worse than an
expensive-but-correct one.

**One k-point along each vacuum axis** (#159). Every k_distance -- human,
ml, llm or heuristic -- and every k_index becomes a mesh in ``kmesh``,
which gives each vacuum axis of the structure it is passed
(``analysis.geometry.vacuum_axes``) exactly one division and builds the
ladder from the periodic axes alone, so the mesh, the ladder and the
reverse lookup (``_with_k_index``) agree. A cell with vacuum along all
three axes is sampled at Gamma only, the one rung of its ladder. The
caller passes the cell the mesh is for (the simulated cell,
``system.cell``, in ``service/_step_kpoints.py``), and the vacuum is read
from that cell,
never from the dimensionality label, which calls bulk graphite '2d'. An
explicit human ``k_grid`` is kept as given, vacuum axes included.

**Coupled axes alike** (#163). ``kmesh`` also gives the lattice vectors
the structure's point group mixes the same count, the larger one, so every
k_distance and k_index mesh is one the point group maps onto itself, and
the ladder's rungs are the distinct meshes of that kind. The structure is
the simulated cell with its species, spin classes included, so the point
group is the one pw.x keeps for a collinear run and contains the one it
keeps for a noncollinear one. An explicit human ``k_grid`` is kept as
given here too, invariant or not.

Not attempted here (and not claimed to be): the ``is_metal``-based
k_distance *ceiling* and the rung-cannot-exceed-the-ladder clamp
(goldilocks-core-design.md:1732-1817) both gate the future ML tier's
output, not this heuristic's.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field
from pymatgen.core import Structure

from goldilocks_core.advisors.occupations import OccupationsDecision
from goldilocks_core.advisors.species_split import split_species
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
    Source,
    Warning,
    field_sources_from,
)

if TYPE_CHECKING:
    from goldilocks_core.ml.predict import MlPrediction

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
    see ``_with_k_index`` below. ``None`` when no rung has the resolved
    mesh -- typically an explicit ``k_grid`` off the ladder -- not a
    failure."""
    k_distance_interval: tuple[float, float | None] | None = None
    """The matched rung's own ``KMeshEntry.k_distance_interval``, carried
    alongside ``k_index`` whenever a rung is known. Exists so a caller
    resolving via ``k_index`` -- where ``k_distance`` itself is
    deliberately ``None``, a whole interval maps to one rung, not one
    point -- still has something honest to show instead of nothing at
    all; showing a fabricated single point (e.g. the interval midpoint)
    would be worse than showing no number. The upper bound is ``None``
    only for rung 1, the Gamma-only mesh, whose interval is open above."""


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
    ``k_grid``/``k_distance``, not a variant of either. The rungs are the
    distinct meshes the structure's point group maps onto itself, so they
    are numbered differently from goldilocks-data's ``ladder_1based.v2``
    wherever the point group mixes lattice vectors of different length
    (#163)."""
    k_distance: float | None = Field(default=None, gt=0)
    shift: tuple[_ZeroOrOne, _ZeroOrOne, _ZeroOrOne] | None = None
    """QE's own K_POINTS automatic card requires each shift component to
    be exactly 0 or 1 (INPUT_PW.txt, checked 2026-09-14) -- any other
    integer used to be accepted and written verbatim."""


class KSamplingLlmInput(LlmInput):
    k_distance: float | None = Field(default=None, gt=0)


@dataclass(frozen=True, slots=True)
class KSamplingProtocol:
    """A task protocol's k-point spacing for one of its steps (#174), such
    as the dos task's denser nscf mesh: ranked below the request's own mesh
    settings and above QRF95 and the llm. Not a request value, so its
    record reads ``heuristic``, with ``note`` naming the protocol as its
    ``source_note``."""

    k_distance: float
    """1/angstrom, as ``KSamplingHumanInput.k_distance``."""
    note: str


def k_sampling(
    is_metal: FieldState[Metallicity],
    structure: Structure,
    occupations: FieldState[OccupationsDecision] | None = None,
    human: KSamplingHumanInput | None = None,
    llm: KSamplingLlmInput | None = None,
    protocol: KSamplingProtocol | None = None,
) -> FieldState[KSamplingDecision]:
    """The mesh from a human ``k_grid``, else a human ``k_index``, else the
    k_distance of the first tier that gives one (``_k_distance_tier``):
    human, the step's ``protocol``, QRF95, llm, then the ``is_metal``
    heuristic."""
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
        return Resolved(
            _with_k_index(decision, structure), _provenance(human, None, "k_grid")
        )
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
        return Resolved(decision, _provenance(human, None, "k_index"))
    tier = _k_distance_tier(is_metal, structure, human, llm, protocol)
    if isinstance(tier, Blocked):
        return tier
    k_distance, shift, provenance = tier
    return Resolved(
        _with_k_index(_from_k_distance(structure, k_distance, shift), structure),
        provenance,
    )


def _k_distance_tier(
    is_metal: FieldState[Metallicity],
    structure: Structure,
    human: KSamplingHumanInput,
    llm: KSamplingLlmInput,
    protocol: KSamplingProtocol | None,
) -> tuple[float, tuple[int, int, int] | None, Provenance] | Blocked:
    """The k_distance of the first tier that gives one, in rank order --
    human, the step's ``protocol``, QRF95, llm, the ``is_metal`` heuristic --
    with the shift and the provenance that tier applies. Only the llm tier
    reads no human shift. A protocol skips QRF95, which is trained on scf
    meshes (#174). ``Blocked`` when the heuristic decides and ``is_metal``
    is blocked."""
    if human.k_distance is not None:
        return human.k_distance, human.shift, _provenance(human, None, "k_distance")
    if protocol is not None:
        return (
            protocol.k_distance,
            human.shift,
            _provenance(human, None, "k_distance", source_note=protocol.note),
        )
    prediction = _ml_k_distance(structure)
    if prediction is not None:
        return (
            float(prediction.value),
            human.shift,
            _provenance(
                human,
                None,
                "k_distance",
                decided="ml",
                model=prediction.provenance.model,
                citations=prediction.provenance.citations,
            ),
        )
    if llm.k_distance is not None:
        return llm.k_distance, None, _provenance(None, llm, "k_distance")
    if isinstance(is_metal, Blocked):
        return Blocked(by=is_metal)
    k_distance = (
        _NON_METAL_K_DISTANCE
        if is_metal.ok and is_metal.value == "non_metal"
        else _METAL_K_DISTANCE
    )
    return (
        k_distance,
        human.shift,
        _provenance(human, None, "k_distance", derived_from=("is_metal",)),
    )


def _provenance(
    human: KSamplingHumanInput | None,
    llm: KSamplingLlmInput | None,
    mesh_from: str,
    *,
    decided: Source = "heuristic",
    **provenance: Any,
) -> Provenance:
    """The provenance of a decision whose mesh came from the setting
    ``mesh_from`` (``k_grid``, ``k_index`` or ``k_distance``), read from the
    inputs its branch applied (#173): ``mesh_from`` and ``shift`` each
    ``human`` or ``llm`` where that input set it, otherwise ``mesh_from``
    the tier that decided it (``decided``) and ``shift`` the heuristic
    Gamma default. ``provenance`` sets the rest: the model's evidence for
    QRF95's k_distance, ``derived_from`` for the heuristic's, which follows
    ``is_metal``, and the protocol's ``source_note`` for a protocol's."""
    field_sources = field_sources_from(
        human, llm, fields=(mesh_from, "shift"), decided={mesh_from: decided}
    )
    return Provenance.from_field_sources(field_sources, **provenance)


def _ml_k_distance(structure: Structure) -> MlPrediction | None:
    """QRF95's prediction of a raw k-distance (v2 epic 11, #11; #92), whose
    ``provenance`` the record carries with QRF95's own interval in its details
    (#156), or ``None`` if it
    is not set up (its model asset is not installed or fails verification),
    in which case ``k_sampling()`` falls back to its next tier, as
    ``analysis/is_metal.py``'s ``_ml_is_metal`` does. A model that is set up
    but fails to load or predict raises, failing ``k_sampling()`` itself
    (#130).

    QRF95 reads the cell's chemistry, so it gets ``structure`` with one
    plain element per site (``split_species`` without classes): the species
    of a spin-polarized cell carry their starting-magnetization class as
    ``Species.spin``, a pw.x species label the model was not trained on,
    and its prediction moved with them (LiFePO4's afm-1 by 11%, #170)."""
    from goldilocks_core.ml.predict import MlModelUnavailable, predict_k_distance

    try:
        return predict_k_distance(split_species(structure, None))
    except MlModelUnavailable:
        return None


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
    on a rung of this structure's own ladder? Every rung of
    ``build_gamma_kmesh_entries`` is a distinct mesh (its own docstring),
    so at most one entry can match and the first hit is authoritative. A
    mesh from a k_distance is built the way the rungs are, coupled axes
    alike (#163). Leaves ``decision`` untouched (``k_index`` stays
    ``None``) when no rung has the mesh: an explicit ``k_grid`` off the
    ladder, such as one that gives coupled axes different counts."""
    for entry in build_gamma_kmesh_entries(structure):
        if entry.mesh == decision.mesh:
            return replace(
                decision,
                k_index=entry.kindex,
                k_distance_interval=entry.k_distance_interval,
            )
    return decision
