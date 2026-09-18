"""list_magnetic_orderings: enumerate, and optionally mMACE-rank, magnetic
orderings for one structure (v2, #87).

Deliberately not part of ``advise()``/``generate()``: those assume one
already-decided ``relabeled_structure`` per calculation, but this is a
listing capability a caller consults *before* deciding which ordering(s)
to actually generate input files for -- "give me FM and AFM input files"
and "let mMACE recommend one" both start here, then feed a chosen
candidate's structure back into ``advisors/magnetic_config.py``'s existing
overrides (``human.starting_magnetization``, or the structure itself) the
normal way. See issue #87 for the full three-layer design (enumerate ->
optionally rank -> generate) this is the first layer of.

Enumeration never needs mMACE (it is
``advisors/magnetic_config.enumerate_magnetic_orderings``, the same
enumlib-backed path the existing heuristic tier already uses) and always
succeeds in the sense of returning at least the plain ferromagnetic
candidate. Ranking is opt-in and can fail independently (missing
``magnetism`` extra, unconfigured checkpoint) without failing the listing
itself -- a caller who asked for ranking still gets every candidate back,
just unranked, with a warning naming why.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from goldilocks_core.advisors.magnetic_config import (
    enumerate_magnetic_orderings,
    magnetic_elements_in,
)
from goldilocks_core.advisors.magnetic_ordering_ml import (
    MagneticOrderingMlUnavailable,
    rank_orderings,
)
from goldilocks_core.resolution import Warning

if TYPE_CHECKING:
    from pymatgen.core import Structure


@dataclass(frozen=True, slots=True)
class MagneticOrderingCandidate:
    """One listed candidate, ranked or not."""

    label: str
    structure: Structure
    natoms: int
    energy_per_atom_ev: float | None
    status: str | None
    is_recommended: bool


@dataclass(frozen=True, slots=True)
class MagneticOrderingsReport:
    candidates: tuple[MagneticOrderingCandidate, ...]
    ranked: bool
    warnings: tuple[Warning, ...] = ()


def _unranked(
    candidates: tuple[tuple[str, Structure], ...],
) -> tuple[MagneticOrderingCandidate, ...]:
    return tuple(
        MagneticOrderingCandidate(
            label=label,
            structure=candidate,
            natoms=len(candidate),
            energy_per_atom_ev=None,
            status=None,
            is_recommended=False,
        )
        for label, candidate in candidates
    )


def list_magnetic_orderings(
    structure: Structure, *, rank_with_mmace: bool = False, device: str = "cpu"
) -> MagneticOrderingsReport:
    """List every magnetic-ordering candidate for ``structure``.

    ``rank_with_mmace=True`` additionally relaxes every candidate's moments
    on the mMACE potential energy surface and marks the lowest
    energy-per-atom candidate as ``is_recommended``. If the ``magnetism``
    extra or its checkpoint is not configured, the candidates are still
    returned (unranked), with a warning naming why -- ranking is an
    enhancement to this listing, never a precondition for it.
    """
    candidates = enumerate_magnetic_orderings(
        structure, magnetic_elements_in(structure)
    )

    if not rank_with_mmace:
        return MagneticOrderingsReport(candidates=_unranked(candidates), ranked=False)

    try:
        ranked = rank_orderings(candidates, device=device)
    except MagneticOrderingMlUnavailable as error:
        return MagneticOrderingsReport(
            candidates=_unranked(candidates),
            ranked=False,
            warnings=(
                Warning(
                    code="magnetic.ordering_ranking_unavailable",
                    level="warning",
                    category="magnetic",
                    message=str(error),
                ),
            ),
        )

    by_label = {result.label: result for result in ranked.candidates}
    return MagneticOrderingsReport(
        candidates=tuple(
            MagneticOrderingCandidate(
                label=label,
                structure=candidate,
                natoms=len(candidate),
                energy_per_atom_ev=by_label[label].energy_per_atom_ev,
                status=by_label[label].status,
                is_recommended=label == ranked.winner.label,
            )
            for label, candidate in candidates
        ),
        ranked=True,
    )
