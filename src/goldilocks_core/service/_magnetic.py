"""The simulated cell and the pseudopotentials chosen for it, each decided
once (#128).

These decisions depend on one another in a fixed order:

1. the functional and ``magnetic_cell`` -- spin polarization, spin-orbit
   coupling, and the requested magnetic ordering applied to the structure,
   which is the calculation's only enumlib call;
2. pseudopotential table selection, which reads the functional and the
   cell's spin-orbit switch (the relativistic requirement);
3. ``magnetic_config``, the starting moments on the cell, scaled by the
   selected pseudopotentials' z_valence.

``symmetry_eff`` is the cell's own symmetry. Every later decision -- the
electron count, bands, k-points, convergence, resources, the gate and the
writer -- reads the same cell object (``MagneticCell.structure``), and the
input structure stops here: no count can describe a cell other than the
one written. The magnetic decision used to run twice, once before table
selection and once after, enumerating each time (#184).
"""

from __future__ import annotations

from dataclasses import dataclass

from pymatgen.core import Structure

from goldilocks_core.advisors.functional import functional
from goldilocks_core.advisors.magnetic_config import (
    MagneticCell,
    MagneticConfigFacts,
    magnetic_cell,
    magnetic_config,
)
from goldilocks_core.analysis.symmetry import SymmetryFacts, symmetry
from goldilocks_core.assets.store import AssetStore
from goldilocks_core.resolution import Blocked, FieldState, blocked_by
from goldilocks_core.service._analysis import AnalysisFacts
from goldilocks_core.service._pseudo import PseudoAdvice, resolve_pseudopotentials
from goldilocks_core.service._system_overrides import SystemOverrides


@dataclass(frozen=True, slots=True)
class MagneticAdvice:
    """The decisions above, in order; ``system_advice`` copies them onto
    ``SystemAdvice``."""

    functional: FieldState[str]
    cell: FieldState[MagneticCell]
    symmetry_eff: FieldState[SymmetryFacts]
    pseudo: PseudoAdvice
    magnetic: FieldState[MagneticConfigFacts]


def magnetic_advice(
    structure: Structure,
    analysis: AnalysisFacts,
    overrides: SystemOverrides,
    *,
    store: AssetStore | None,
    fetch_missing: bool,
) -> MagneticAdvice:
    """The functional, the simulated cell, its symmetry, the
    pseudopotentials and the starting moments for ``structure``."""
    functional_state = functional(overrides.functional, overrides.functional_llm)
    cell = magnetic_cell(
        structure, analysis.is_magnetic, overrides.magnetic, overrides.magnetic_llm
    )
    if cell.ok:
        pseudo = resolve_pseudopotentials(
            elements=set(analysis.composition.value.elements),
            functional=functional_state.value,
            spin_orbit_enabled=cell.value.spin_orbit_enabled,
            table_id=overrides.pseudo_table_id,
            store=store,
            fetch_missing=fetch_missing,
        )
        symmetry_eff = symmetry(cell.value.structure)
    else:
        cause = Blocked(by=blocked_by(cell))
        pseudo = PseudoAdvice(table=cause, metadata=cause, relativistic=cause)
        symmetry_eff = cause
    return MagneticAdvice(
        functional=functional_state,
        cell=cell,
        symmetry_eff=symmetry_eff,
        pseudo=pseudo,
        magnetic=_seeded(cell, pseudo, analysis, overrides),
    )


def _seeded(
    cell: FieldState[MagneticCell],
    pseudo: PseudoAdvice,
    analysis: AnalysisFacts,
    overrides: SystemOverrides,
) -> FieldState[MagneticConfigFacts]:
    """``magnetic_config`` on the selected table's z_valence. A table that
    is not installed yet (``Unavailable``, a normal preview) leaves the
    flat default fractions; a selection that failed outright (``Blocked``:
    no compatible table exists, e.g. spin-orbit coupling on a lanthanide
    that only SSSP covers) blocks the decision instead of serving
    uncalibrated seeds as if they were good."""
    if isinstance(pseudo.metadata, Blocked):
        return Blocked(by=pseudo.metadata)
    z_valences = (
        {
            item.element: item.z_valence
            for item in pseudo.metadata.value
            if item.element is not None and item.z_valence is not None
        }
        if pseudo.metadata.ok
        else None
    )
    return magnetic_config(cell, analysis.needs_soc, z_valences, overrides.magnetic)
