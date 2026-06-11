from __future__ import annotations

from goldilocks_core.advise.types import Protocol, SmearingDecision
from goldilocks_core.analyse.structure import StructureAnalysis
from goldilocks_core.intent import CalculationIntent

_HINT_METHOD = "smearing_method"
_HINT_WIDTH_EV = "smearing_width_ev"

# metallicity values that force smearing (guardrail: cannot be overridden to fixed)
_FORCE_SMEARING = {"metallic", "likely_metallic", "unknown"}


def advise_smearing(
    analysis: StructureAnalysis,
    intent: CalculationIntent,
    protocol: Protocol,
) -> SmearingDecision:
    """Return a SmearingDecision from structure analysis and calculation intent.

    Method selection (implicit, best-available):
      Metallicity is always heuristic in Phase 1 (element-based).
      Phase 2 will add ML metallicity classification.

    Guardrail: metallic / likely_metallic / unknown metallicity always uses
    smearing regardless of hints. Insulating structures default to fixed
    occupations but can be overridden via hints['smearing_method'].
    """
    hints = intent.hints

    # Resolve width up-front so all rationale strings can include it.
    # user_hint wins over protocol default.
    if _HINT_WIDTH_EV in hints:
        width_ev: float | None = float(hints[_HINT_WIDTH_EV])
        width_src = "user_hint"
    else:
        width_ev = protocol.smearing_width_ev
        width_src = f"{protocol.name!r} protocol"

    if analysis.metallicity in _FORCE_SMEARING:
        # Determine smearing method
        if _HINT_METHOD in hints:
            method = str(hints[_HINT_METHOD])
            method_src = "user_hint"
            provenance = "user_hint"
        else:
            method = "marzari_vanderbilt"
            method_src = "default"
            provenance = "heuristic"

        _valid_methods = {"marzari_vanderbilt", "methfessel_paxton", "fermi_dirac", "gaussian"}
        if method not in _valid_methods:
            raise ValueError(
                f"Unknown smearing method {method!r}. Valid: {sorted(_valid_methods)}"
            )

        rationale = (
            f"Heuristic metallicity={analysis.metallicity!r} "
            f"(source: {analysis.metallicity_source!r}) → smearing required "
            f"(guardrail: metals and unknowns cannot use fixed occupations). "
            f"Method: {method!r} ({method_src}), "
            f"width: {width_ev:.4f} eV ({width_src})."
        )
        return SmearingDecision(
            use_smearing=True,
            method=method,  # type: ignore[arg-type]
            width_ev=width_ev,
            provenance=provenance,  # type: ignore[arg-type]
            rationale=rationale,
        )

    # insulating / likely_insulating
    if _HINT_METHOD in hints:
        method = str(hints[_HINT_METHOD])
        _valid_methods = {"marzari_vanderbilt", "methfessel_paxton", "fermi_dirac", "gaussian"}
        if method not in _valid_methods:
            raise ValueError(
                f"Unknown smearing method {method!r}. Valid: {sorted(_valid_methods)}"
            )
        rationale = (
            f"Heuristic metallicity={analysis.metallicity!r} suggests fixed occupations, "
            f"but user_hint requests smearing ({method!r}, {width_ev:.4f} eV)."
        )
        return SmearingDecision(
            use_smearing=True,
            method=method,  # type: ignore[arg-type]
            width_ev=width_ev,
            provenance="user_hint",
            rationale=rationale,
        )

    return SmearingDecision(
        use_smearing=False,
        method=None,
        width_ev=None,
        provenance="heuristic",
        rationale=(
            f"Heuristic metallicity={analysis.metallicity!r} "
            f"(source: {analysis.metallicity_source!r}) → fixed occupations recommended. "
            f"Use hints[{_HINT_METHOD!r}] to override if smearing is needed."
        ),
    )
