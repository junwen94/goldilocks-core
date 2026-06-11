from __future__ import annotations

from goldilocks_core.advise.types import CutoffDecision, PseudoSelection
from goldilocks_core.analyse.structure import StructureAnalysis
from goldilocks_core.intent import CalculationIntent

_EV_PER_RY: float = 13.605693122994

# Accuracy-tier defaults for norm-conserving (NC) pseudos.
# ecutrho = 4 × ecutwfc is the NC rule (charge density = 4× wavefunction cutoff).
# Values are conservative PseudoDojo-compatible estimates; actual pseudo metadata
# will raise this floor in Phase 2.
_ECUTWFC_RY: dict[str, float] = {
    "fast":     40.0,
    "balanced": 60.0,
    "accurate": 80.0,
}
_NC_RHO_FACTOR: int = 4


def advise_basis(
    analysis: StructureAnalysis,
    intent: CalculationIntent,
    pseudos: list[PseudoSelection],
) -> CutoffDecision:
    """Return a CutoffDecision (eV) from the accuracy tier and pseudo metadata.

    Phase 1: uses accuracy-tier defaults as the floor. If any PseudoSelection
    carries non-zero wavefunction_cutoff_ev (resolved by Select in Phase 2),
    takes the element-wise maximum.

    Hints 'ecutwfc_ev' and 'ecutrho_ev' override everything (both required).
    """
    hints = intent.hints

    if hints.ecutwfc_ev is not None and hints.ecutrho_ev is not None:
        wfc_ev = hints.ecutwfc_ev
        rho_ev = hints.ecutrho_ev
        return CutoffDecision(
            wavefunction_cutoff_ev=wfc_ev,
            density_cutoff_ev=rho_ev,
            provenance="user_hint",
            rationale=(
                f"cutoffs overridden by user_hint: "
                f"ecutwfc={wfc_ev:.1f} eV, ecutrho={rho_ev:.1f} eV"
            ),
        )

    base_ry = _ECUTWFC_RY[intent.accuracy]
    accuracy_wfc_ev = base_ry * _EV_PER_RY
    accuracy_rho_ev = accuracy_wfc_ev * _NC_RHO_FACTOR

    # Raise floor from per-element pseudo cutoffs when available
    pseudo_wfc_floor = max((p.wavefunction_cutoff_ev for p in pseudos), default=0.0)
    pseudo_rho_floor = max((p.density_cutoff_ev for p in pseudos), default=0.0)

    wfc_ev = max(accuracy_wfc_ev, pseudo_wfc_floor)
    rho_ev = max(accuracy_rho_ev, pseudo_rho_floor)

    rationale = (
        f"accuracy={intent.accuracy!r} → "
        f"ecutwfc={base_ry:.0f} Ry ({accuracy_wfc_ev:.1f} eV), "
        f"ecutrho={base_ry * _NC_RHO_FACTOR:.0f} Ry ({accuracy_rho_ev:.1f} eV)"
    )
    if pseudo_wfc_floor > accuracy_wfc_ev:
        rationale += f"; raised by pseudo metadata to {wfc_ev:.1f} eV"

    return CutoffDecision(
        wavefunction_cutoff_ev=wfc_ev,
        density_cutoff_ev=rho_ev,
        provenance="heuristic",
        rationale=rationale,
    )
