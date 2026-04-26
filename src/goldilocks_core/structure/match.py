from __future__ import annotations

from dataclasses import dataclass

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

_MATCHER = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)


@dataclass
class MatchResult:
    formula: str
    spacegroup: str | None
    source: str
    url: str
    matched: bool
    score: float | None


def run_matcher(
    query: Structure,
    candidates: list[tuple[Structure, dict]],
) -> list[MatchResult]:
    """Run StructureMatcher against a list of (structure, metadata) candidates.

    metadata keys: formula, spacegroup, source, url
    Returns results sorted by: exact matches first, then by RMS score.
    """
    results: list[MatchResult] = []

    for candidate_structure, meta in candidates:
        matched = False
        score: float | None = None
        try:
            matched = bool(_MATCHER.fit(query, candidate_structure))
            if matched:
                rms = _MATCHER.get_rms_dist(query, candidate_structure)
                if rms is not None:
                    score = float(rms[0])
        except Exception:
            pass

        results.append(
            MatchResult(
                formula=meta.get("formula", ""),
                spacegroup=meta.get("spacegroup"),
                source=meta.get("source", ""),
                url=meta.get("url", ""),
                matched=matched,
                score=score,
            )
        )

    results.sort(
        key=lambda r: (not r.matched, r.score if r.score is not None else 999.0)
    )
    return results
