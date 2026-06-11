from __future__ import annotations

from goldilocks_core.advise.types import SpinDecision
from goldilocks_core.analyse.structure import StructureAnalysis
from goldilocks_core.intent import CalculationIntent, ParameterHints

_VALID_TREATMENTS = frozenset({"non_magnetic", "collinear", "non_collinear", "non_collinear_soc"})

# Initial magnetic moments in μB — FM start, conservative high values so QE
# doesn't collapse to non-magnetic.  Mirrors aiida-quantumespresso's
# magnetization.yaml (magmom column).  Generate stage divides by z_val to
# obtain QE starting_magnetization ∈ [-1, 1].
_ELEMENT_INITIAL_MAG: dict[str, float] = {
    # 3d transition metals
    "Sc": 5.0, "Ti": 5.0, "V":  5.0, "Cr": 5.0, "Mn": 5.0,
    "Fe": 5.0, "Co": 5.0, "Ni": 5.0,
    # 4d transition metals (commonly magnetic)
    "Nb": 5.0, "Mo": 5.0, "Tc": 5.0, "Ru": 5.0, "Rh": 5.0,
    # 5d transition metals
    "Hf": 5.0, "Ta": 5.0, "W":  5.0, "Re": 5.0, "Os": 5.0,
    "Ir": 5.0, "Pt": 5.0,
    # lanthanides
    "La": 5.0, "Ce": 5.0, "Pr": 7.0, "Nd": 7.0, "Pm": 7.0,
    "Sm": 7.0, "Eu": 7.0, "Gd": 5.0, "Tb": 7.0, "Dy": 7.0,
    "Ho": 7.0, "Er": 7.0, "Tm": 7.0,
    # actinides
    "Ac": 5.0, "Th": 5.0, "Pa": 5.0, "U":  5.0, "Np": 5.0,
    "Pu": 7.0,
}
_FALLBACK_MAG: float = 5.0       # μB for unlisted magnetic elements
_DEFAULT_NONMAG_MAG: float = 0.1 # μB for non-magnetic species in non-collinear

_ML_INSTALL_TIP = (
    "Install goldilocks-core[magnetic] and pass a MagneticClassifier to "
    "analyze_structure() to enable ML-based magnetic prediction."
)


def _parse_mag_hint(raw: object) -> dict[str, float]:
    """Parse initial_magnetization hint into {element: μB} dict.

    Accepts:
      - dict as-is: {"Fe": 3.0}
      - "Fe:3.0" or "Fe:3.0,Ni:1.5" (CLI-friendly colon-separated format)
    """
    if isinstance(raw, dict):
        return {str(k): float(v) for k, v in raw.items()}
    if isinstance(raw, str):
        result: dict[str, float] = {}
        for token in raw.replace(";", ",").split(","):
            token = token.strip()
            if not token:
                continue
            if ":" not in token:
                raise ValueError(
                    f"Cannot parse initial_magnetization token {token!r}. "
                    "Expected format: El:value  e.g. Fe:3.0 or Fe:3.0,Ni:1.5"
                )
            el, _, val = token.partition(":")
            result[el.strip()] = float(val.strip())
        return result
    raise ValueError(
        f"initial_magnetization hint must be a dict or 'El:val' string, got {type(raw).__name__!r}"
    )


def _initial_magnetization(
    elements: list[str],
    magnetic_elements: list[str],
    hints: ParameterHints,
    include_all: bool = False,
) -> dict[str, float] | None:
    """Return initial magnetic moments (μB) per element.

    include_all=True (non-collinear): ALL species are included.  Non-magnetic
    species get _DEFAULT_NONMAG_MAG so QE doesn't start from zero and collapse.
    include_all=False (collinear): only magnetic species; QE default (0) for others.
    """
    if hints.initial_magnetization is not None:
        return _parse_mag_hint(hints.initial_magnetization)
    if not magnetic_elements:
        return None
    if include_all:
        return {
            el: _ELEMENT_INITIAL_MAG.get(el, _FALLBACK_MAG)
            if el in magnetic_elements
            else _DEFAULT_NONMAG_MAG
            for el in elements
        }
    return {el: _ELEMENT_INITIAL_MAG.get(el, _FALLBACK_MAG) for el in magnetic_elements}


def _angles(elements: list[str]) -> dict[str, float]:
    """Return angle1 or angle2 dict with all elements set to 0.0 (FM along z)."""
    return {el: 0.0 for el in elements}


def _heuristic_treatment(analysis: StructureAnalysis) -> str:
    """Return the heuristic-only spin treatment label, ignoring ML predictions."""
    if not analysis.magnetic_elements:
        return "non_magnetic"
    if analysis.soc_relevant:
        return "non_collinear_soc"
    return "collinear"


def advise_spin(
    analysis: StructureAnalysis,
    intent: CalculationIntent,
) -> SpinDecision:
    """Return a SpinDecision from structure analysis and calculation intent.

    Method selection (implicit, best-available):
      1. user_hint  — 'spin_treatment' in hints overrides everything
      2. ML         — analysis.magnetic_prediction not None (mMACE classifier)
      3. heuristic  — element-lookup fallback (always available)

    The rationale field explains which method was used, the reasoning, and
    (for heuristic) how to enable ML classification.
    """
    hints = intent.hints

    # --- user_hint overrides all ---
    if hints.spin_treatment is not None:
        treatment = hints.spin_treatment
        if treatment not in _VALID_TREATMENTS:
            raise ValueError(
                f"Unknown spin treatment {treatment!r}. Valid: {sorted(_VALID_TREATMENTS)}"
            )
        is_noncolin = treatment in ("non_collinear", "non_collinear_soc")
        initial_mag = (
            None if treatment == "non_magnetic"
            else _initial_magnetization(
                analysis.elements, analysis.magnetic_elements, hints,
                include_all=is_noncolin,
            )
        )
        return SpinDecision(
            treatment=treatment,  # type: ignore[arg-type]
            initial_magnetization=initial_mag,
            angle1=_angles(analysis.elements) if is_noncolin else None,
            angle2=_angles(analysis.elements) if is_noncolin else None,
            provenance="user_hint",
            rationale=f"User override: spin_treatment={treatment!r}.",
        )

    # --- ML path (mMACE classifier ran during analyse stage) ---
    if analysis.magnetic_prediction is not None:
        ml_pred = analysis.magnetic_prediction
        soc_upgraded = False

        if ml_pred == "non_magnetic":
            treatment_str = "non_magnetic"
            initial_mag = None
        elif ml_pred == "non_collinear":
            treatment_str = "non_collinear_soc" if analysis.soc_relevant else "non_collinear"
            soc_upgraded = analysis.soc_relevant
            initial_mag = _initial_magnetization(
                analysis.elements, analysis.magnetic_elements, hints, include_all=True
            )
        else:  # "collinear"
            treatment_str = "collinear"
            initial_mag = _initial_magnetization(
                analysis.elements, analysis.magnetic_elements, hints, include_all=False
            )

        heuristic_str = _heuristic_treatment(analysis)
        if heuristic_str == treatment_str:
            comparison = f"heuristic agrees ({heuristic_str!r})"
        else:
            comparison = f"heuristic would give {heuristic_str!r} — ML overrides"

        conf_str = (
            f"{analysis.magnetic_confidence:.2f}"
            if analysis.magnetic_confidence is not None
            else "n/a"
        )
        soc_note = (
            f"; SOC-relevant heavy elements {analysis.heavy_elements!r} → upgraded to non_collinear_soc"
            if soc_upgraded
            else ""
        )
        rationale = (
            f"ML classifier (mMACE backbone, confidence={conf_str}): "
            f"predicted {ml_pred!r} → {treatment_str!r}{soc_note}. "
            f"{comparison}."
        )
        is_noncolin = treatment_str in ("non_collinear", "non_collinear_soc")
        return SpinDecision(
            treatment=treatment_str,  # type: ignore[arg-type]
            initial_magnetization=initial_mag,
            angle1=_angles(analysis.elements) if is_noncolin else None,
            angle2=_angles(analysis.elements) if is_noncolin else None,
            provenance="ML",
            rationale=rationale,
        )

    # --- heuristic fallback ---
    has_mag = bool(analysis.magnetic_elements)

    if not has_mag:
        return SpinDecision(
            treatment="non_magnetic",
            initial_magnetization=None,
            angle1=None,
            angle2=None,
            provenance="heuristic",
            rationale=(
                "Heuristic: no magnetic elements in composition → non_magnetic. "
                f"{_ML_INSTALL_TIP}"
            ),
        )

    if analysis.soc_relevant:
        return SpinDecision(
            treatment="non_collinear_soc",
            initial_magnetization=_initial_magnetization(
                analysis.elements, analysis.magnetic_elements, hints, include_all=True
            ),
            angle1=_angles(analysis.elements),
            angle2=_angles(analysis.elements),
            provenance="heuristic",
            rationale=(
                f"Heuristic: magnetic elements {analysis.magnetic_elements!r} + "
                f"SOC-relevant heavy elements {analysis.heavy_elements!r} "
                f"→ non_collinear_soc. "
                f"{_ML_INSTALL_TIP}"
            ),
        )

    return SpinDecision(
        treatment="collinear",
        initial_magnetization=_initial_magnetization(
            analysis.elements, analysis.magnetic_elements, hints, include_all=False
        ),
        angle1=None,
        angle2=None,
        provenance="heuristic",
        rationale=(
            f"Heuristic: magnetic elements {analysis.magnetic_elements!r}, "
            f"no SOC-relevant species → collinear spin-polarised (FM start). "
            f"{_ML_INSTALL_TIP}"
        ),
    )
