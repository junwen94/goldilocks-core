"""Direct database search via OPTIMADE.

Mirrors the approach in goldilocks-api/app/routes/structure_match.py.
No API keys required.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx
from pymatgen.core import Composition

_TIMEOUT = 30.0
_MAX_RESULTS = 20


@dataclass
class SearchResult:
    source: str
    formula: str
    spacegroup: str | None
    entry_id: str | None
    url: str


# ---------------------------------------------------------------------------
# Formula normalisation — OPTIMADE Hill order, no count-1
# ---------------------------------------------------------------------------

def normalise_formula(formula: str) -> str:
    """Reduce to Hill-ordered formula for OPTIMADE filter.

    e.g. "O4Fe3" → "Fe3O4", "FeO" → "FeO"
    """
    try:
        comp = Composition(formula).reduced_composition
        elements = sorted(
            comp.elements,
            key=lambda e: (e.symbol != "C", e.symbol != "H", e.symbol),
        )
        parts = []
        for el in elements:
            count = int(comp[el])
            parts.append(el.symbol + (str(count) if count != 1 else ""))
        return "".join(parts)
    except Exception:
        return formula


# ---------------------------------------------------------------------------
# Per-database async queries (MP, MC, NOMAD, JARVIS)
# ---------------------------------------------------------------------------

async def _query_mp(
    formula: str, client: httpx.AsyncClient
) -> tuple[list[SearchResult], str | None]:
    """Materials Project via OPTIMADE (no API key required)."""
    params = {
        "filter": f'chemical_formula_reduced = "{formula}"',
        "response_fields": "id,chemical_formula_reduced",
        "page_limit": _MAX_RESULTS,
    }
    try:
        resp = await client.get(
            "https://optimade.materialsproject.org/v1/structures",
            params=params,
        )
        if resp.status_code >= 500:
            return [], None  # server-side error — treat as empty, not user-facing error
        resp.raise_for_status()
        results = []
        for item in resp.json().get("data", []):
            mp_id = item.get("id", "")
            attrs = item.get("attributes", {})
            if not mp_id:
                continue
            results.append(SearchResult(
                source="Materials Project",
                formula=attrs.get("chemical_formula_reduced", ""),
                spacegroup=None,  # not available via OPTIMADE endpoint
                entry_id=mp_id,
                url=f"https://next-gen.materialsproject.org/materials/{mp_id}",
            ))
        return results, None
    except Exception as exc:  # noqa: BLE001
        return [], str(exc)


async def _query_mc(
    formula: str, client: httpx.AsyncClient
) -> tuple[list[SearchResult], str | None]:
    """Materials Cloud MC3D via OPTIMADE."""
    params = {
        "filter": f'chemical_formula_reduced = "{formula}"',
        "response_fields": "id,chemical_formula_reduced,_mcloud_mc3d_id",
        "page_limit": _MAX_RESULTS,
    }
    try:
        resp = await client.get(
            "https://optimade.materialscloud.org/main/mc3d-pbesol-v2/structures",
            params=params,
        )
        if resp.status_code >= 500:
            return [], None
        resp.raise_for_status()
        results = []
        for item in resp.json().get("data", []):
            attrs = item.get("attributes", {})
            mc3d_id = attrs.get("_mcloud_mc3d_id")
            if not mc3d_id:
                continue
            results.append(SearchResult(
                source="Materials Cloud",
                formula=attrs.get("chemical_formula_reduced", ""),
                spacegroup=None,
                entry_id=str(mc3d_id),
                url=f"https://mc3d.materialscloud.org/details/{mc3d_id}/pbesol-v2",
            ))
        return results, None
    except Exception as exc:  # noqa: BLE001
        return [], str(exc)


async def _query_nomad(
    formula: str, client: httpx.AsyncClient
) -> tuple[list[SearchResult], str | None]:
    """NOMAD via OPTIMADE."""
    params = {
        "filter": f'chemical_formula_reduced = "{formula}"',
        "response_fields": (
            "id,chemical_formula_reduced,"
            "_nmd_entry_page_url,"
            "_nmd_results_material_symmetry_space_group_symbol"
        ),
        "page_limit": _MAX_RESULTS,
    }
    try:
        resp = await client.get(
            "https://nomad-lab.eu/prod/v1/optimade/structures",
            params=params,
        )
        if resp.status_code >= 500:
            return [], None  # NOMAD chokes on some formulas — treat as empty
        resp.raise_for_status()
        results = []
        for item in resp.json().get("data", []):
            attrs = item.get("attributes", {})
            url = attrs.get("_nmd_entry_page_url") or ""
            if not url:
                continue
            results.append(SearchResult(
                source="NOMAD",
                formula=attrs.get("chemical_formula_reduced", ""),
                spacegroup=attrs.get(
                    "_nmd_results_material_symmetry_space_group_symbol"
                ),
                entry_id=item.get("id"),
                url=url,
            ))
        return results, None
    except Exception as exc:  # noqa: BLE001
        return [], str(exc)


async def _query_jarvis(
    formula: str, client: httpx.AsyncClient
) -> tuple[list[SearchResult], str | None]:
    """JARVIS-DFT via OPTIMADE (no local download required)."""
    params = {
        "filter": f'chemical_formula_reduced = "{formula}"',
        "response_fields": "id,chemical_formula_reduced,_jarvis_jid,_jarvis_spg_symbol,_jarvis_reference",
        "page_limit": _MAX_RESULTS,
    }
    try:
        resp = await client.get(
            "https://jarvis.nist.gov/optimade/jarvisdft/v1/structures/",
            params=params,
        )
        if resp.status_code >= 500:
            return [], None
        resp.raise_for_status()
        results = []
        for item in resp.json().get("data", []):
            attrs = item.get("attributes", {})
            jid = attrs.get("_jarvis_jid") or item.get("id", "")
            if not jid:
                continue
            ref = attrs.get("_jarvis_reference") or ""
            url = ref if ref else f"https://www.ctcms.nist.gov/~knc6/static/JARVIS-DFT/{jid}"
            results.append(SearchResult(
                source="JARVIS",
                formula=attrs.get("chemical_formula_reduced", ""),
                spacegroup=attrs.get("_jarvis_spg_symbol"),
                entry_id=jid,
                url=url,
            ))
        return results, None
    except Exception as exc:  # noqa: BLE001
        return [], str(exc)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def search_databases(
    formula: str,
) -> tuple[list[SearchResult], dict[str, str]]:
    """Query MP, Materials Cloud, NOMAD, JARVIS (all parallel) for *formula*.

    Returns (results, errors) where errors maps source name → error string.
    Only unexpected failures appear in errors; 500s from servers are silently
    treated as empty results (matching goldilocks-api behaviour).
    """
    normalised = normalise_formula(formula)

    async def _gather() -> tuple[list[SearchResult], dict[str, str]]:
        async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
            (mp_res, mp_err), (mc_res, mc_err), (nomad_res, nomad_err), (jarvis_res, jarvis_err) = (
                await asyncio.gather(
                    _query_mp(normalised, client),
                    _query_mc(normalised, client),
                    _query_nomad(normalised, client),
                    _query_jarvis(normalised, client),
                )
            )
        errors: dict[str, str] = {}
        if mp_err:
            errors["Materials Project"] = mp_err
        if mc_err:
            errors["Materials Cloud"] = mc_err
        if nomad_err:
            errors["NOMAD"] = nomad_err
        if jarvis_err:
            errors["JARVIS"] = jarvis_err
        return mp_res + mc_res + nomad_res + jarvis_res, errors

    return asyncio.run(_gather())
