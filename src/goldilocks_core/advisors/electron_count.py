"""electron_count: nelec, from the structure's composition and the
selected pseudopotentials' z_valence.

New in v2 (v2 epic 7, #7). No advisor computed this before --
`advisors/nbnd.py` (v2 epic 6, #6) already documented `nelec` as "a
required, plain input... no advisor computes this yet", naming
`advisors/pseudo_selection.py`'s eventual materialize step as the
missing piece. This is that piece: `nelec = sum(z_valence[element] *
count[element] for element in structure)`, summed over the structure's
actual composition (not just the set of distinct elements), matching
how a real UPF pseudopotential reports its own valence electron count
per atom -- ``PseudoMetadata.z_valence``, already parsed from UPF
headers or provider sidecars (v2 epic 3, #4).
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import Field
from pymatgen.core import Structure

from goldilocks_core.assets.pseudopotentials.upf import PseudoMetadata
from goldilocks_core.inputs.overrides import HumanInput
from goldilocks_core.resolution import (
    Blocked,
    FieldState,
    Provenance,
    Resolved,
    blocked_by,
)


@dataclass(frozen=True, slots=True)
class ElectronCountDecision:
    nelec: float


class ElectronCountHumanInput(HumanInput):
    nelec: float | None = Field(default=None, gt=0)


def electron_count(
    structure: Structure,
    pseudos: FieldState[tuple[PseudoMetadata, ...]],
    human: ElectronCountHumanInput | None = None,
) -> FieldState[ElectronCountDecision]:
    human = human or ElectronCountHumanInput()

    if human.nelec is not None:
        return Resolved(ElectronCountDecision(human.nelec), Provenance(source="human"))

    if isinstance(pseudos, Blocked):
        return Blocked(by=pseudos)
    if not pseudos.ok:
        return Blocked(by=blocked_by(pseudos))

    z_valence_by_element = {
        metadata.element: metadata.z_valence for metadata in pseudos.value
    }
    counts = structure.composition.get_el_amt_dict()

    missing = sorted(
        element for element in counts if z_valence_by_element.get(element) is None
    )
    if missing:
        return Blocked(by=f"no z_valence for {', '.join(missing)}")

    nelec = sum(
        z_valence_by_element[element] * count for element, count in counts.items()
    )
    return Resolved(ElectronCountDecision(nelec), Provenance(source="heuristic"))
