from __future__ import annotations

from typing import Literal

from goldilocks_core.advise.types import Protocol
from goldilocks_core.analyse.structure import StructureAnalysis
from goldilocks_core.intent import CalculationIntent

_PROTOCOLS: dict[str, Protocol] = {
    "stringent": Protocol("stringent", 0.0125, 0.1701, 0.10),
    "balanced":  Protocol("balanced",  0.0200, 0.2721, 0.15),
    "fast":      Protocol("fast",      0.0275, 0.3742, 0.30),
}

_ACCURACY_TO_PROTOCOL: dict[str, str] = {
    "accurate": "stringent",
    "balanced": "balanced",
    "fast":     "fast",
}


def select_protocol(
    analysis: StructureAnalysis,
    intent: CalculationIntent,
) -> tuple[Protocol, Literal["heuristic"]]:
    """Return the sampling protocol and its provenance (always heuristic).

    Metals with lanthanides or actinides are forced to Stringent regardless
    of accuracy tier. Insulators are capped at Balanced.
    """
    base_name = _ACCURACY_TO_PROTOCOL[intent.accuracy]

    is_metal = analysis.metallicity in {"metallic", "likely_metallic", "unknown"}
    has_f = analysis.contains_lanthanides or analysis.contains_actinides
    is_insulator = analysis.metallicity in {"insulating", "likely_insulating"}

    if is_metal and has_f and base_name != "stringent":
        return _PROTOCOLS["stringent"], "heuristic"

    if is_insulator and base_name == "stringent":
        return _PROTOCOLS["balanced"], "heuristic"

    return _PROTOCOLS[base_name], "heuristic"
