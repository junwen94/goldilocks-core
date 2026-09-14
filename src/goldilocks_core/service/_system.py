"""Phase 2 of the v2 orchestrator (v2 epic 8, #8): system-level advisors.

Mirrors ``system_settings.SystemSettings``'s own field list one layer
up the pipeline: ``SystemAdvice`` holds the same eight decisions
(``pseudo`` composes the two pseudopotential-related ones -- see
``_pseudo.PseudoAdvice``'s own docstring on why), still as
``FieldState`` (pre-``checks.check_all``), not yet unwrapped to plain
values. See ``service/__init__.py`` for the phase boundaries this split
follows.

**Two-pass ``magnetic_config``, per its own docstring**: it cannot call
pseudopotential selection itself (selection needs ``spin_orbit_enabled``,
which ``magnetic_config`` decides -- an import cycle), so this module
calls it twice: once with ``z_valences=None`` to get a provisional
``spin_orbit_enabled`` for building pseudopotential requirements, and
again after a table is chosen, with the real per-element ``z_valence``
values, for the calibrated ``starting_magnetization`` that actually
lands in ``SystemAdvice.magnetic``.
"""

from __future__ import annotations

from dataclasses import dataclass

from pymatgen.core import Structure

from goldilocks_core.advisors.boundary import BoundaryFacts, boundary
from goldilocks_core.advisors.cutoffs import CutoffsDecision, cutoffs
from goldilocks_core.advisors.electron_count import (
    ElectronCountDecision,
    electron_count,
)
from goldilocks_core.advisors.functional import functional
from goldilocks_core.advisors.hubbard_u import HubbardUDecision, hubbard_u
from goldilocks_core.advisors.magnetic_config import (
    MagneticConfigFacts,
    magnetic_config,
)
from goldilocks_core.advisors.vdw_method import VdwFacts, vdw_method
from goldilocks_core.assets.store import AssetStore
from goldilocks_core.resolution import Blocked, FieldState, blocked_by
from goldilocks_core.service._analysis import AnalysisFacts
from goldilocks_core.service._pseudo import PseudoAdvice, resolve_pseudopotentials
from goldilocks_core.service._system_overrides import SystemOverrides


@dataclass(frozen=True, slots=True)
class SystemAdvice:
    functional: FieldState[str]
    pseudo: PseudoAdvice
    magnetic: FieldState[MagneticConfigFacts]
    cutoffs: FieldState[CutoffsDecision]
    electron_count: FieldState[ElectronCountDecision]
    vdw: FieldState[VdwFacts]
    boundary: FieldState[BoundaryFacts]
    hubbard: FieldState[HubbardUDecision]

    def field_states(self) -> tuple[FieldState[object], ...]:
        """Everything except ``magnetic`` -- ``checks.check_all`` folds
        that in itself; see its own docstring on why a caller must not
        repeat it here."""
        return (
            self.functional,
            self.pseudo.table,
            self.pseudo.metadata,
            self.cutoffs,
            self.electron_count,
            self.vdw,
            self.boundary,
            self.hubbard,
        )

    def records(self) -> dict[str, FieldState[object]]:
        return {
            "functional": self.functional,
            "pseudo_table": self.pseudo.table,
            "pseudopotentials": self.pseudo.metadata,
            "magnetic": self.magnetic,
            "cutoffs": self.cutoffs,
            "electron_count": self.electron_count,
            "vdw": self.vdw,
            "boundary": self.boundary,
            "hubbard": self.hubbard,
        }


def system_advice(
    structure: Structure,
    analysis: AnalysisFacts,
    overrides: SystemOverrides,
    *,
    store: AssetStore | None,
    fetch_missing: bool,
) -> SystemAdvice:
    functional_state = functional(overrides.functional, overrides.functional_llm)
    functional_value = functional_state.value

    magnetic_provisional = magnetic_config(
        structure,
        analysis.is_magnetic,
        analysis.needs_soc,
        None,
        overrides.magnetic,
        overrides.magnetic_llm,
    )

    if magnetic_provisional.ok:
        pseudo = resolve_pseudopotentials(
            elements=set(analysis.composition.value.elements),
            functional=functional_value,
            spin_orbit_enabled=magnetic_provisional.value.spin_orbit_enabled,
            table_id=overrides.pseudo_table_id,
            store=store,
            fetch_missing=fetch_missing,
        )
    else:
        cause = blocked_by(magnetic_provisional)
        pseudo = PseudoAdvice(table=Blocked(by=cause), metadata=Blocked(by=cause))

    if pseudo.metadata.ok:
        z_valences = {
            item.element: item.z_valence
            for item in pseudo.metadata.value
            if item.element is not None and item.z_valence is not None
        }
        magnetic_state = magnetic_config(
            structure,
            analysis.is_magnetic,
            analysis.needs_soc,
            z_valences,
            overrides.magnetic,
            overrides.magnetic_llm,
        )
    else:
        magnetic_state = magnetic_provisional

    return SystemAdvice(
        functional=functional_state,
        pseudo=pseudo,
        magnetic=magnetic_state,
        cutoffs=cutoffs(pseudo.metadata, pseudo.table, overrides.cutoffs),
        electron_count=electron_count(
            structure, pseudo.metadata, overrides.electron_count
        ),
        vdw=vdw_method(
            analysis.geometry, functional_value, overrides.vdw, overrides.vdw_llm
        ),
        boundary=boundary(analysis.geometry),
        hubbard=hubbard_u(
            structure,
            analysis.needs_correlation,
            functional_value,
            overrides.hubbard,
            overrides.hubbard_llm,
        ),
    )
