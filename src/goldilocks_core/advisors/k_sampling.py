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
This module's job is only to guarantee a working, zero-ML-dependency
path exists -- the actual ML tier (goldilocks-data's ladder plus a
k_index/k_distance model) is v2 epic 11, and is stubbed here the same
way every other advisor in this codebase stubs its ml tier.

``occupations`` is accepted as an explicit input -- not because this
heuristic tier uses it (it does not; both branches below are a flat
k_distance regardless of smearing width) -- but because the point of
this epic's per-step ordering is to make sigma an *explicit* input to
k_sampling rather than an implicit shared condition (the goldilocks
dataset itself was generated at fixed ``cold``/0.01 Ry smearing while
only ``k_index`` was scanned, so pretending the two are decoupled would
be dishonest). Once epic 11 wires in a real k_distance/k_index model,
that model's own contract is expected to actually consume this.

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

from dataclasses import dataclass
from typing import Annotated

from pydantic import Field
from pymatgen.core import Structure

from goldilocks_core.advisors.occupations import OccupationsDecision
from goldilocks_core.analysis.is_metal import Metallicity
from goldilocks_core.inputs.overrides import HumanInput, LlmInput
from goldilocks_core.kmesh import k_distance_to_mesh
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
        "both k_grid and k_distance were given; k_grid wins outright and "
        "k_distance is ignored."
    ),
)

WARNING_CATALOGUE = (GRID_AND_DISTANCE_WARNING,)
"""Every warning code this module can emit -- ``capabilities.py``'s
``warnings[]`` catalogue aggregates one of these tuples per advisor."""


@dataclass(frozen=True, slots=True)
class KSamplingDecision:
    mesh: tuple[int, int, int]
    shift: tuple[int, int, int]
    k_distance: float | None
    warnings: tuple[Warning, ...] = ()


class KSamplingHumanInput(HumanInput):
    k_grid: tuple[_PositiveInt, _PositiveInt, _PositiveInt] | None = None
    """Each axis count must be >= 1 (#-- found in the epic 5/6/7 delivery
    -layer audit): a 0 or negative entry used to be accepted, resolved,
    and written verbatim into the K_POINTS card, silently telling QE to
    sample zero or a negative number of points along that axis."""
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
        warnings = (GRID_AND_DISTANCE_WARNING,) if human.k_distance is not None else ()
        decision = KSamplingDecision(
            mesh=human.k_grid,
            shift=human.shift or _GAMMA_SHIFT,
            k_distance=None,
            warnings=warnings,
        )
        return Resolved(decision, Provenance(source="human"))
    if human.k_distance is not None:
        return Resolved(
            _from_k_distance(structure, human.k_distance, human.shift),
            Provenance(source="human"),
        )
    ml_value: float | None = None  # no ml model wired yet; stubbed until epic 11
    if ml_value is not None:
        return Resolved(
            _from_k_distance(structure, ml_value, None), Provenance(source="ml")
        )
    if llm.k_distance is not None:
        return Resolved(
            _from_k_distance(structure, llm.k_distance, None), Provenance(source="llm")
        )

    if isinstance(is_metal, Blocked):
        return Blocked(by=is_metal)
    if is_metal.ok and is_metal.value == "non_metal":
        return Resolved(
            _from_k_distance(structure, _NON_METAL_K_DISTANCE, human.shift),
            Provenance(source="heuristic"),
        )
    return Resolved(
        _from_k_distance(structure, _METAL_K_DISTANCE, human.shift),
        Provenance(source="heuristic"),
    )


def _from_k_distance(
    structure: Structure, k_distance: float, shift: tuple[int, int, int] | None
) -> KSamplingDecision:
    return KSamplingDecision(
        mesh=k_distance_to_mesh(structure, k_distance),
        shift=shift or _GAMMA_SHIFT,
        k_distance=k_distance,
    )
