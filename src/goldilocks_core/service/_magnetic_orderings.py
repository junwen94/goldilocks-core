"""list_magnetic_orderings: enumerate, and optionally mMACE-rank, magnetic
orderings for one structure (v2, #87).

Deliberately not part of ``advise()``/``generate()``: those assume one
already-decided ``relabeled_structure`` per calculation, but this is a
listing capability a caller consults *before* deciding which ordering(s)
to actually generate input files for -- "give me FM and AFM input files"
and "let mMACE recommend one" both start here. Each listed candidate
(``candidate_to_json``) already carries the ``structure_content``/
``overrides`` a caller feeds straight back to ``/run`` as a brand new
structure input to generate that exact candidate's files -- see
``_bundle_inputs`` for why an AFM candidate needs an explicit
``starting_magnetization`` override alongside its structure. See issue
#87 for the full three-layer design (enumerate -> optionally rank ->
generate) this is the first layer of.

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

from dataclasses import dataclass, field
from typing import Any

from pymatgen.core import Structure

from goldilocks_core.advisors.magnetic_config import (
    afm_unavailable_warning,
    enumerate_magnetic_orderings,
    magnetic_elements_in,
    starting_magnetization_for,
)
from goldilocks_core.advisors.magnetic_ordering_ml import (
    MagneticOrderingMlUnavailable,
    rank_orderings,
)
from goldilocks_core.resolution import Warning


@dataclass(frozen=True, slots=True)
class MagneticOrderingCandidate:
    """One listed candidate, ranked or not."""

    label: str
    structure: Structure
    natoms: int
    energy_per_atom_ev: float | None
    status: str | None
    is_recommended: bool
    structure_content: str
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MagneticOrderingsReport:
    candidates: tuple[MagneticOrderingCandidate, ...]
    ranked: bool
    warnings: tuple[Warning, ...] = ()


def candidate_to_json(candidate: MagneticOrderingCandidate) -> dict[str, Any]:
    """The JSON-safe projection of one candidate -- shared by
    ``cli/_magnetic_orderings.py`` and ``server/_handlers.py`` so both
    transports report the same shape.

    ``structure_content``/``structure_format`` plus ``overrides`` are
    everything a caller needs to actually generate DFT input files for
    *this exact* candidate: feed them straight back as a new top-level
    structure + overrides to ``/run`` (or ``generate()`` directly), no
    label lookup or re-enumeration required. This is deliberately a fresh
    structure input, not a ``relabeled_structure`` override plumbed
    through ``advise()`` for the *original* structure -- an AFM candidate
    can be a different cell (a `MagneticStructureEnumerator` supercell)
    from the one the caller started with, so it has to run its own
    ``advise()``/pseudopotential selection from scratch, same as any other
    structure upload. See ``_bundle_inputs`` for why the FM candidate
    needs no overrides at all while every AFM candidate needs an explicit
    ``starting_magnetization`` (CIF round-tripping silently drops
    ``Species.spin`` -- the sign information that distinguishes the two
    magnetic sublattices -- so it cannot be left for ``/run``'s own
    heuristic to rediscover)."""
    return {
        "label": candidate.label,
        "formula": candidate.structure.composition.reduced_formula,
        "natoms": candidate.natoms,
        "energy_per_atom_ev": candidate.energy_per_atom_ev,
        "status": candidate.status,
        "is_recommended": candidate.is_recommended,
        "structure_content": candidate.structure_content,
        "structure_format": "cif",
        "overrides": candidate.overrides,
    }


def report_to_json(report: MagneticOrderingsReport) -> dict[str, Any]:
    return {
        "ranked": report.ranked,
        "candidates": [candidate_to_json(candidate) for candidate in report.candidates],
        "warnings": [warning.model_dump() for warning in report.warnings],
    }


def _bundle_inputs(label: str, structure: Structure) -> tuple[str, dict[str, Any]]:
    """The ``(structure_content, overrides)`` pair ``candidate_to_json``
    exposes for generating this candidate's own DFT input files.

    The FM candidate is exactly the caller's original structure -- the
    existing ``/run`` pipeline already handles it correctly with no
    overrides at all, so it gets none. Every AFM candidate carries
    per-site ``Species.spin`` that CIF text cannot round-trip (confirmed
    empirically: pymatgen's own CIF writer/reader drops it, leaving every
    site's spin ``None``); left alone, re-loading this CIF would silently
    resolve a *ferromagnetic* ``starting_magnetization`` despite the
    species labels still looking AFM-split. Computing the signed fraction
    from the original, spin-decorated structure and remapping it onto the
    labels the CIF will actually carry once re-parsed (by site order, not
    by label string -- pymatgen's CIF writer renumbers labels, e.g.
    ``Fe1``/``Fe2`` in memory becomes ``Fe0``/``Fe1`` on disk) keeps the
    two magnetic sublattices correctly opposed end to end."""
    cif = structure.to(fmt="cif")
    if label == "fm":
        return cif, {}
    roundtripped = Structure.from_str(cif, fmt="cif")
    fractions_by_original_label = starting_magnetization_for(structure)
    return cif, {
        "spin_polarized": True,
        "starting_magnetization": {
            reloaded_site.label: fractions_by_original_label[original_site.label]
            for original_site, reloaded_site in zip(
                structure, roundtripped, strict=True
            )
        },
    }


def _unranked(
    candidates: tuple[tuple[str, Structure], ...],
) -> tuple[MagneticOrderingCandidate, ...]:
    candidates_out = []
    for label, candidate in candidates:
        structure_content, overrides = _bundle_inputs(label, candidate)
        candidates_out.append(
            MagneticOrderingCandidate(
                label=label,
                structure=candidate,
                natoms=len(candidate),
                energy_per_atom_ev=None,
                status=None,
                is_recommended=False,
                structure_content=structure_content,
                overrides=overrides,
            )
        )
    return tuple(candidates_out)


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

    Also warns (rather than silently listing "fm" alone, indistinguishable
    from "this structure genuinely has no AFM candidates") whenever
    ``structure`` actually has a magnetic element but AFM enumeration
    itself could not run or find one -- the common real case being
    enumlib (``enum.x``/``multienum.x``) not installed.
    """
    magnetic_elements = magnetic_elements_in(structure)
    candidates, afm_reason = enumerate_magnetic_orderings(structure, magnetic_elements)
    warnings = (
        (afm_unavailable_warning(afm_reason),)
        if afm_reason is not None and magnetic_elements
        else ()
    )

    if not rank_with_mmace:
        return MagneticOrderingsReport(
            candidates=_unranked(candidates), ranked=False, warnings=warnings
        )

    try:
        ranked = rank_orderings(candidates, device=device)
    except MagneticOrderingMlUnavailable as error:
        return MagneticOrderingsReport(
            candidates=_unranked(candidates),
            ranked=False,
            warnings=(
                *warnings,
                Warning(
                    code="magnetic.ordering_ranking_unavailable",
                    level="warning",
                    category="magnetic",
                    message=str(error),
                ),
            ),
        )

    by_label = {result.label: result for result in ranked.candidates}
    candidates_out = []
    for label, candidate in candidates:
        structure_content, overrides = _bundle_inputs(label, candidate)
        candidates_out.append(
            MagneticOrderingCandidate(
                label=label,
                structure=candidate,
                natoms=len(candidate),
                energy_per_atom_ev=by_label[label].energy_per_atom_ev,
                status=by_label[label].status,
                is_recommended=label == ranked.winner.label,
                structure_content=structure_content,
                overrides=overrides,
            )
        )
    return MagneticOrderingsReport(
        candidates=tuple(candidates_out), ranked=True, warnings=warnings
    )
