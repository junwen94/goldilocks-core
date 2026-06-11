"""Heuristic van der Waals correction adviser (code-agnostic).

Method names are physics labels; per-code translation lives in the
Generate layer:

  d3   → QE "grimme-d3"           (DFT-D3, zero damping)
  d3bj → QE "grimme-d3bj"         (DFT-D3, Becke-Johnson damping)
  ts   → QE "ts-vdw"              (Tkatchenko-Scheffler)
  mbd  → QE "many-body-dispersion" (Many-Body Dispersion)

Rule:
  Non-3D systems (2D slab, 1D wire, 0D molecule) or any structure
  with explicit vacuum → apply D3BJ (most widely validated for surfaces
  and low-dimensional materials with PBE/PBEsol).
  3D bulk → no correction by default (user can override via hints).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from goldilocks_core.analyse.structure import StructureAnalysis
    from goldilocks_core.intent import CalculationIntent

from goldilocks_core.advise.types import VdwDecision

_DEFAULT_METHOD: Literal["d3bj"] = "d3bj"


def advise_vdw(
    analysis: "StructureAnalysis",
    intent: "CalculationIntent",
) -> VdwDecision:
    """Return a code-agnostic vdW correction decision.

    Priority: user_hint > heuristic.

    Heuristic trigger:
      - ``dimensionality != "3d"``  (slab, wire, molecule)
      - ``has_vacuum``              (explicit vacuum in the cell)
    """
    hints = intent.hints

    # ── User hint override ───────────────────────────────────────────────────
    if "use_vdw" in hints:
        use = bool(hints["use_vdw"])
        method_raw = hints.get("vdw_method", _DEFAULT_METHOD) if use else None
        method: Literal["d3", "d3bj", "ts", "mbd"] | None = method_raw  # type: ignore[assignment]
        return VdwDecision(
            use_vdw=use,
            method=method,
            provenance="user_hint",
            rationale=(
                f"user_hint: use_vdw={use}"
                + (f", method={method}" if use else "")
            ),
        )

    # ── Heuristic ────────────────────────────────────────────────────────────
    non_3d = analysis.dimensionality != "3d"
    has_vac = analysis.has_vacuum

    if non_3d or has_vac:
        trigger = (
            f"{analysis.dimensionality}/{analysis.system_type}"
            if non_3d
            else "explicit vacuum"
        )
        return VdwDecision(
            use_vdw=True,
            method=_DEFAULT_METHOD,
            provenance="heuristic",
            rationale=(
                f"{trigger} → dispersion correction applied (D3BJ).  "
                "Interlayer / surface binding is dominated by dispersion; "
                "D3BJ is well-validated for PBE/PBEsol with surfaces and "
                "low-dimensional materials."
            ),
        )

    return VdwDecision(
        use_vdw=False,
        method=None,
        provenance="heuristic",
        rationale=(
            "3D bulk → vdW correction not applied by default.  "
            "If the material is layered (e.g. graphite, MoS₂ bulk) or a "
            "molecular crystal, set hint use_vdw=True."
        ),
    )
