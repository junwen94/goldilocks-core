from __future__ import annotations

from pathlib import Path

from goldilocks_core.advise.types import PseudoSelection
from goldilocks_core.analyse.structure import StructureAnalysis
from goldilocks_core.intent import CalculationIntent

_HINT_PSEUDO_FAMILY = "pseudo_family"
_HINT_PSEUDO_DIR = "pseudo_dir"

_SR_TAG = "/SR/"
_FR_TAG = "/FR/"

_EV_PER_RY: float = 13.605693122994


def _to_fr_family(family: str) -> str:
    """Replace the SR component of a PseudoDojo family label with FR."""
    return family.replace(_SR_TAG, _FR_TAG)


def _is_fr_family(family: str) -> bool:
    return _FR_TAG in family


def _resolve_from_registry(
    elements: list[str],
    family: str,
    pseudo_dir: Path,
    provenance: str,
) -> list[PseudoSelection]:
    """Resolve PseudoSelections from a local UPF directory."""
    from goldilocks_core.pseudo.registry import (
        filter_by_element,
        filter_by_relativistic,
        load_pseudo_metadata,
    )

    all_metadata = load_pseudo_metadata(pseudo_dir)
    relativistic = "full" if _is_fr_family(family) else "scalar"

    selections: list[PseudoSelection] = []
    for el in elements:
        candidates = filter_by_element(all_metadata, el)
        candidates = filter_by_relativistic(candidates, relativistic)

        if not candidates:
            # Fall back to unfiltered element match if relativistic filter yields nothing
            candidates = filter_by_element(all_metadata, el)

        if not candidates:
            # No UPF found: return placeholder and let caller handle the gap
            selections.append(PseudoSelection(
                element=el,
                family=family,
                filename="",
                path=None,
                wavefunction_cutoff_ev=0.0,
                density_cutoff_ev=0.0,
                provenance=provenance,  # type: ignore[arg-type]
            ))
            continue

        meta = candidates[0]
        wfc_ry, rho_ry = _extract_cutoffs_ry(meta)

        selections.append(PseudoSelection(
            element=el,
            family=family,
            filename=meta.filename,
            path=Path(meta.filepath) if meta.filepath else None,
            wavefunction_cutoff_ev=wfc_ry * _EV_PER_RY,
            density_cutoff_ev=rho_ry * _EV_PER_RY,
            provenance=provenance,  # type: ignore[arg-type]
        ))

    return selections


def _extract_cutoffs_ry(meta: object) -> tuple[float, float]:
    """Extract (ecutwfc_ry, ecutrho_ry) from a PseudoMetadata object.

    Tries SSSP recommended cutoffs first, then UPF header pseudo_info,
    then falls back to 0.0 (basis.py accuracy defaults take over).
    """
    # SSSP: sssp_recommended_cutoff = {"ecutwfc_ry": ..., "ecutrho_ry": ...}
    sssp_cutoff = getattr(meta, "sssp_recommended_cutoff", None)
    if isinstance(sssp_cutoff, dict):
        wfc = sssp_cutoff.get("ecutwfc_ry")
        rho = sssp_cutoff.get("ecutrho_ry")
        if wfc is not None and rho is not None:
            return float(wfc), float(rho)

    # PseudoDojo / generic UPF header: pseudo_info["Suggested cutoff for wfc and rho"]
    pseudo_info = getattr(meta, "pseudo_info", {}) or {}
    suggested = pseudo_info.get("Suggested cutoff for wfc and rho", {})
    if isinstance(suggested, dict):
        wfc = suggested.get("ecutwfc_ry")
        rho = suggested.get("ecutrho_ry")
        if wfc is not None and rho is not None:
            return float(wfc), float(rho)

    # rho only (some UPF v1 headers)
    rho_only = pseudo_info.get("rho_cutoff")
    if rho_only is not None:
        rho = float(rho_only)
        return rho / 4.0, rho  # NC approximation: ecutwfc ≈ ecutrho / 4

    return 0.0, 0.0


def advise_pseudos(
    analysis: StructureAnalysis,
    intent: CalculationIntent,
) -> list[PseudoSelection]:
    """Return one PseudoSelection per element in the structure.

    Determines the pseudo family label, upgrading SR → FR when SOC-relevant
    heavy elements are present and the intent family is SR.

    If hint 'pseudo_dir' points to a local UPF directory, resolves actual
    filenames, paths, and cutoffs via the pseudo registry. Otherwise returns
    placeholder PseudoSelections (path=None, cutoffs=0.0) — file resolution
    is deferred to the Select stage or aiida-pseudo at runtime.
    """
    hints = intent.hints
    elements = sorted(set(analysis.elements))

    if _HINT_PSEUDO_FAMILY in hints:
        family = str(hints[_HINT_PSEUDO_FAMILY])
        provenance = "user_hint"
    elif analysis.soc_relevant and _SR_TAG in intent.pseudo_family:
        family = _to_fr_family(intent.pseudo_family)
        provenance = "heuristic"
    else:
        family = intent.pseudo_family
        provenance = "heuristic"

    # Resolve pseudo directory: hint > bundled data > aiida-pseudo placeholder
    if _HINT_PSEUDO_DIR in hints:
        pseudo_dir_path = Path(str(hints[_HINT_PSEUDO_DIR]))
        return _resolve_from_registry(elements, family, pseudo_dir_path, provenance)

    # Try bundled data (ships with the package)
    try:
        from goldilocks_core.data import pseudo_dir as bundled_pseudo_dir
        pseudo_dir_path = bundled_pseudo_dir(family)
        return _resolve_from_registry(elements, family, pseudo_dir_path, provenance)
    except FileNotFoundError:
        pass

    # aiida-pseudo mode: family label only, file resolution deferred to runtime
    return [
        PseudoSelection(
            element=el,
            family=family,
            filename="",
            path=None,
            wavefunction_cutoff_ev=0.0,
            density_cutoff_ev=0.0,
            provenance=provenance,  # type: ignore[arg-type]
        )
        for el in elements
    ]
