"""Pseudopotential table selection + asset-store materialization, fully
encapsulated behind one function (v2 epic 8, #8).

**A gap discovered while building this that neither epic 5 nor epic 7
closed**: turning a chosen ``PseudoTable`` into the actual per-element
``PseudoMetadata`` used for generation needs (a) resolving the table's
installed asset off the ``AssetStore``, (b) loading its full manifest,
and (c) narrowing that to the structure's own elements. Step (c) is
``advisors/pseudo_selection.py``'s new ``select_metadata_for_elements``
(added alongside this module); v1's ``pseudo/source.py``/``selection.py``
was deliberately *not* reused for this -- it carries both the exact
``StopIteration``-prone bug epic 5 already fixed at this layer, and a
cross-table ranking algorithm that no longer applies once table choice
is settled first (see that function's own docstring).

Kept as its own module, not folded into ``_system.py``: this is the one
place in the whole pipeline that touches the asset store/filesystem, and
collapsing table selection + requirements-building + installation +
manifest-loading + per-element narrowing behind ``resolve_pseudopotentials``
keeps every asset-store-shaped type (``AssetStore``, ``AssetNotInstalled``,
``InstalledAsset``, the registry/importers modules) out of ``_system.py``'s
own import surface entirely.
"""

from __future__ import annotations

from dataclasses import dataclass

from goldilocks_core.advisors.pseudo_selection import (
    pseudo_requirements,
    select_metadata_for_elements,
    select_pseudopotential_table,
)
from goldilocks_core.assets.pseudopotentials.importers import load_installed_table
from goldilocks_core.assets.pseudopotentials.registry import PseudoTable, load_tables
from goldilocks_core.assets.pseudopotentials.upf import PseudoMetadata
from goldilocks_core.assets.runtime import install as install_assets
from goldilocks_core.assets.store import AssetNotInstalled, AssetStore
from goldilocks_core.resolution import (
    Blocked,
    FieldState,
    Provenance,
    Resolved,
    Unavailable,
    blocked_by,
)
from goldilocks_core.types import RelativisticTreatment


@dataclass(frozen=True, slots=True)
class PseudoAdvice:
    """Both pseudopotential-related ``FieldState``s, composed here (not
    as two loose fields on ``SystemAdvice``) so that ``PseudoTable``/
    ``PseudoMetadata`` -- and every asset-store type behind them -- stay
    entirely inside this module's own import surface. This project's
    complexity gate (``scripts/check_complexity.py``) resolves re-exports
    back to their true origin, so ``_system.py`` would still be charged
    for these types even importing them indirectly; composing them into
    one record defined and consumed from here is what actually keeps
    them off ``_system.py``'s ledger.

    ``relativistic`` is the *confirmed* treatment -- backed by an actual
    selected table, not merely the requested one -- so it ``Blocked``s
    exactly when ``table`` itself could not be selected at all (v2 epic
    9, #9: the SOC-on-Ce acceptance scenario needs this as its own named,
    independently-checkable field, not only buried inside ``table``'s own
    ``.value.relativistic``, which is unreadable once ``table`` blocks)."""

    table: FieldState[PseudoTable]
    metadata: FieldState[tuple[PseudoMetadata, ...]]
    relativistic: FieldState[RelativisticTreatment]


def resolve_pseudopotentials(
    *,
    elements: set[str],
    functional: str,
    spin_orbit_enabled: bool,
    table_id: str | None,
    store: AssetStore | None,
    fetch_missing: bool,
) -> PseudoAdvice:
    """Select a table (via the same ``PseudoRequirements`` shape
    ``spin_orbit_enabled`` feeds, per ``advisors/pseudo_selection.py``),
    then materialize it into per-element metadata. ``fetch_missing``
    governs only the one real download this pipeline can trigger --
    P8's "no sneaky downloads" -- and still degrades to ``Unavailable``
    rather than raising when it is ``False`` and the table is missing.
    """
    requirements = pseudo_requirements(
        functional, spin_orbit_enabled=spin_orbit_enabled
    )
    table_state = select_pseudopotential_table(
        load_tables(), table_id=table_id, elements=elements, requirements=requirements
    )
    if not table_state.ok:
        cause = Blocked(by=blocked_by(table_state))
        return PseudoAdvice(table=table_state, metadata=cause, relativistic=cause)

    # A table was actually selected, so the relativistic treatment it
    # implies is confirmed, not merely requested -- true regardless of
    # whether the asset itself is installed yet below.
    relativistic_state: FieldState[RelativisticTreatment] = Resolved(
        requirements.relativistic, Provenance(source="heuristic")
    )

    table = table_state.value
    active_store = store or AssetStore()
    try:
        installed = active_store.resolve_spec(table.asset)
    except AssetNotInstalled:
        if not fetch_missing:
            return PseudoAdvice(
                table=table_state,
                metadata=Unavailable(
                    reason=(
                        f"pseudopotential table {table.id!r} is not installed; "
                        f"run 'goldilocks assets install {table.id}' or retry "
                        "with fetch_missing"
                    )
                ),
                relativistic=relativistic_state,
            )
        install_assets(table.asset.id, store=active_store)
        try:
            installed = active_store.resolve_spec(table.asset)
        except AssetNotInstalled as error:
            return PseudoAdvice(
                table=table_state,
                metadata=Unavailable(
                    reason=(
                        f"pseudopotential table {table.id!r} is still not "
                        f"installed after fetching: {error}"
                    )
                ),
                relativistic=relativistic_state,
            )

    full_metadata = load_installed_table(installed, table=table)
    return PseudoAdvice(
        table=table_state,
        metadata=select_metadata_for_elements(full_metadata, elements),
        relativistic=relativistic_state,
    )
