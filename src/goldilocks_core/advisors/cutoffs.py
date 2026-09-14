"""cutoffs: ecutwfc/ecutrho, from the selected pseudopotentials' own
published recommendations.

New in v2 (v2 epic 7, #7). No advisor computed this before -- v1 picked
these cutoffs *inside* `generation/qe/scf.py`
(`max(p["ecutwfc_ry"] for p in selected_pseudos)`, same for `ecutrho_ry`):
a scientific choice made in the layer this rewrite requires to be pure
translation. Moved here unchanged in substance -- max across the
structure's selected pseudopotentials -- matching aiida-pseudo's own
`RecommendedCutoffMixin.get_recommended_cutoffs()`, which does exactly
this (checked against its source 2026-09-14): look up each present
element's own `cutoff_wfc`/`cutoff_rho`, take the max of each
independently. The two maxima are not required to come from the same
element or preserve any particular ratio to each other -- that is
`aiida-pseudo`'s own behaviour too, not a shortcut taken here.

**Where `ecutrho_ry` comes from when a pseudopotential does not publish
one directly.** SSSP tables give both `ecutwfc_ry`/`ecutrho_ry`
explicitly per element (no derivation needed). This codebase's actual
default provider, PseudoDojo (`advisors/pseudo_selection.py`'s
`requires_sssp`: SSSP is the exception, only for lanthanides/actinides),
distributes only a wavefunction cutoff -- the density cutoff has to be
derived as `ecutwfc * charge_density_dual`. That dual is not invented
here: `PseudoTable.charge_density_dual`
(`assets/pseudopotentials/registry.py:91`) is already a required,
populated field for every PseudoDojo table in the live registry
(`registry.toml`: `4.0` throughout, matching both QE's own general
guidance for norm-conserving pseudopotentials and aiida-pseudo's own
`dual_mapping` for norm-conserving pseudo types) -- this module only
reads it, it does not maintain its own copy.

If a pseudopotential's metadata carries neither an explicit `ecutrho_ry`
nor a table with a `charge_density_dual` to derive one from, that is a
genuine gap this module cannot safely paper over with an invented
number -- it returns `Blocked` naming the element, not a guess.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import Field

from goldilocks_core.assets.pseudopotentials.registry import PseudoTable
from goldilocks_core.assets.pseudopotentials.upf import PseudoMetadata
from goldilocks_core.inputs.overrides import HumanInput
from goldilocks_core.resolution import (
    Blocked,
    FieldState,
    Provenance,
    Resolved,
    Warning,
    blocked_by,
)

WARNING_CATALOGUE = (
    Warning(
        code="cutoffs.ecutrho_derived_from_dual",
        level="info",
        category="cutoffs",
        message=(
            "ecutrho derived as charge_density_dual x ecutwfc for one or more "
            "elements (no explicit value published for that pseudopotential)."
        ),
    ),
)
"""Every warning code this module can emit -- ``capabilities.py``'s
``warnings[]`` catalogue aggregates one of these tuples per advisor. The
real, per-occurrence message (naming the actual element and dual) is
built at the call site in ``_aggregate`` below; this entry is the
generic description of the code, for a caller that has not seen it fire."""


@dataclass(frozen=True, slots=True)
class CutoffsDecision:
    ecutwfc_ry: float
    ecutrho_ry: float
    warnings: tuple[Warning, ...] = ()


class CutoffsHumanInput(HumanInput):
    ecutwfc_ry: float | None = Field(default=None, gt=0)
    ecutrho_ry: float | None = Field(default=None, gt=0)


def cutoffs(
    pseudos: FieldState[tuple[PseudoMetadata, ...]],
    table: FieldState[PseudoTable] | None = None,
    human: CutoffsHumanInput | None = None,
) -> FieldState[CutoffsDecision]:
    human = human or CutoffsHumanInput()

    if human.ecutwfc_ry is not None and human.ecutrho_ry is not None:
        decision = CutoffsDecision(human.ecutwfc_ry, human.ecutrho_ry)
        return Resolved(decision, Provenance(source="human"))

    if isinstance(pseudos, Blocked):
        return Blocked(by=pseudos)
    if not pseudos.ok:
        return Blocked(by=blocked_by(pseudos))

    aggregated = _aggregate(pseudos.value, table)
    if isinstance(aggregated, Blocked):
        return aggregated
    ecutwfc_ry, ecutrho_ry, warnings = aggregated

    if human.ecutwfc_ry is None and human.ecutrho_ry is None:
        decision = CutoffsDecision(ecutwfc_ry, ecutrho_ry, warnings)
        return Resolved(decision, Provenance(source="heuristic"))

    decision = CutoffsDecision(
        ecutwfc_ry=human.ecutwfc_ry if human.ecutwfc_ry is not None else ecutwfc_ry,
        ecutrho_ry=human.ecutrho_ry if human.ecutrho_ry is not None else ecutrho_ry,
        warnings=warnings,
    )
    return Resolved(decision, Provenance(source="human"))


def _aggregate(
    pseudos: tuple[PseudoMetadata, ...], table: FieldState[PseudoTable] | None
) -> tuple[float, float, tuple[Warning, ...]] | Blocked:
    dual = table.value.charge_density_dual if table is not None and table.ok else None
    wfc_values: list[float] = []
    rho_values: list[float] = []
    warnings: list[Warning] = []
    for metadata in pseudos:
        wfc = metadata.cutoffs.get("ecutwfc_ry") if metadata.cutoffs else None
        if wfc is None:
            return Blocked(
                by=f"no recommended ecutwfc for pseudopotential {metadata.element!r}"
            )
        wfc_values.append(wfc)

        rho = metadata.cutoffs.get("ecutrho_ry") if metadata.cutoffs else None
        if rho is None:
            if dual is None:
                return Blocked(
                    by=f"no recommended ecutrho for {metadata.element!r} and no "
                    "charge_density_dual to derive one from"
                )
            rho = wfc * dual
            warnings.append(
                Warning(
                    code="cutoffs.ecutrho_derived_from_dual",
                    level="info",
                    category="cutoffs",
                    message=(
                        f"{metadata.element}: ecutrho derived as {dual}x ecutwfc "
                        "(no explicit value published)"
                    ),
                )
            )
        rho_values.append(rho)
    return max(wfc_values), max(rho_values), tuple(warnings)
