from __future__ import annotations

from goldilocks_core.advise.types import Protocol, SmearingDecision
from goldilocks_core.analyse.structure import StructureAnalysis
from goldilocks_core.intent import CalculationIntent

# metallicity values that force smearing (guardrail: cannot be overridden to fixed)
_FORCE_SMEARING = {"metallic", "likely_metallic", "unknown"}
_VALID_METHODS = frozenset({"marzari_vanderbilt", "methfessel_paxton", "fermi_dirac", "gaussian"})


def advise_smearing(
    analysis: StructureAnalysis,
    intent: CalculationIntent,
    protocol: Protocol,
) -> SmearingDecision:
    """Return a SmearingDecision from structure analysis and calculation intent.

    Guardrail: metallic / likely_metallic / unknown metallicity always uses
    smearing regardless of hints. Insulating structures default to fixed
    occupations but can be overridden via hints.smearing_method.
    """
    hints = intent.hints

    if hints.smearing_width_ev is not None:
        width_ev: float | None = hints.smearing_width_ev
        width_src = "user_hint"
    else:
        width_ev = protocol.smearing_width_ev
        width_src = f"{protocol.name!r} protocol"

    if analysis.metallicity in _FORCE_SMEARING:
        if hints.smearing_method is not None:
            method = hints.smearing_method
            method_src = "user_hint"
            provenance = "user_hint"
        else:
            method = "marzari_vanderbilt"
            method_src = "default"
            provenance = "heuristic"

        if method not in _VALID_METHODS:
            raise ValueError(
                f"Unknown smearing method {method!r}. Valid: {sorted(_VALID_METHODS)}"
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
    if hints.smearing_method is not None:
        method = hints.smearing_method
        if method not in _VALID_METHODS:
            raise ValueError(
                f"Unknown smearing method {method!r}. Valid: {sorted(_VALID_METHODS)}"
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
            "Use hints.smearing_method to override if smearing is needed."
        ),
    )
