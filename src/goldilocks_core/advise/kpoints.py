from __future__ import annotations

from goldilocks_core.advise.types import KPointsDecision, Protocol
from goldilocks_core.analyse.structure import StructureAnalysis
from goldilocks_core.intent import CalculationIntent
from goldilocks_core.kmesh import (
    build_k_distance_intervals,
    generate_candidate_k_distances,
    k_distance_to_mesh,
)

_HINT_KPOINTS_GRID  = "kpoints_grid"
_HINT_KPOINTS_SHIFT = "kpoints_shift"
_HINT_K_DISTANCE    = "k_distance"


def advise_kpoints(
    analysis: StructureAnalysis,
    intent: CalculationIntent,
    protocol: Protocol,
    k_index: int | None = None,
    k_distance_ml: float | None = None,
) -> KPointsDecision:
    """Return a KPointsDecision.

    ML models may predict different metrics; all paths converge on kmesh.py:

    Args:
        k_index: ML-predicted index into the k-distance interval schedule
            (k_index path: build_k_distance_intervals → ivs[k_index-1]).
        k_distance_ml: ML-predicted k_distance (Å⁻¹) directly
            (k_distance path: k_distance_to_mesh).
        If both are None, falls back to accuracy-tier heuristic.
        k_index takes priority over k_distance_ml.
    """
    hints = intent.hints

    # --- grid ---
    if _HINT_KPOINTS_GRID in hints:
        raw = hints[_HINT_KPOINTS_GRID]
        grid: tuple[int, int, int] = (int(raw[0]), int(raw[1]), int(raw[2]))
        provenance = "user_hint"
        rationale = f"k-point grid overridden by user_hint: {grid}"

    elif k_index is not None:
        # ML path A: integer k_index → interval schedule (build_k_distance_intervals)
        candidates = generate_candidate_k_distances(intent.structure)
        ivs = build_k_distance_intervals(intent.structure, candidates)
        resolved = max(1, min(k_index, len(ivs)))
        grid = ivs[resolved - 1][0]
        provenance = "ML"
        rationale = (
            f"ML model (k_index): k_index={resolved} → grid={grid} "
            f"(heuristic would use protocol={protocol.name!r} → "
            f"k_distance={protocol.k_distance} Å⁻¹)."
        )

    elif k_distance_ml is not None:
        # ML path B: predicted k_distance (Å⁻¹) → k_distance_to_mesh
        grid = k_distance_to_mesh(intent.structure, k_distance_ml)
        provenance = "ML"
        rationale = (
            f"ML model (k_distance): predicted {k_distance_ml:.4f} Å⁻¹ → grid={grid} "
            f"(heuristic would use protocol={protocol.name!r} → "
            f"k_distance={protocol.k_distance} Å⁻¹)."
        )

    else:
        # Heuristic path: accuracy tier → k_distance → mesh
        if _HINT_K_DISTANCE in hints:
            k_distance = float(hints[_HINT_K_DISTANCE])
            provenance = "user_hint"
            rationale = (
                f"User override: k_distance={k_distance} Å⁻¹ → grid={{}}"
            )
        else:
            k_distance = protocol.k_distance
            provenance = "heuristic"
        grid = k_distance_to_mesh(intent.structure, k_distance)
        if provenance == "user_hint":
            rationale = rationale.format(grid)
        else:
            rationale = (
                f"Heuristic: protocol={protocol.name!r} (accuracy={intent.accuracy!r}) "
                f"→ k_distance={k_distance} Å⁻¹ → grid={grid}. "
                f"No ML k-points model available."
            )

    # Clamp non-periodic directions to 1 (slab / wire / molecule)
    if not all(analysis.pbc):
        grid = tuple(  # type: ignore[assignment]
            n if periodic else 1
            for n, periodic in zip(grid, analysis.pbc)
        )
        rationale += f"; non-periodic axes clamped → {grid}"

    # --- shift ---
    if _HINT_KPOINTS_SHIFT in hints:
        raw_shift = hints[_HINT_KPOINTS_SHIFT]
        shift: tuple[int, int, int] = (
            int(raw_shift[0]),
            int(raw_shift[1]),
            int(raw_shift[2]),
        )
        provenance = "user_hint"
    else:
        shift = (0, 0, 0)

    return KPointsDecision(
        grid=grid,
        shift=shift,
        provenance=provenance,  # type: ignore[arg-type]
        rationale=rationale,
    )
