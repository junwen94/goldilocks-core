from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pymatgen.core import Structure


@dataclass(frozen=True, slots=True)
class CalculationIntent:
    """First-class input object for the goldilocks-core pipeline.

    Args:
        structure: Input crystal structure.
        code: DFT code.
            Supported: ``"qe"``.
            Planned: ``"vasp"``.
        task: Calculation type. Follows pw.x ``calculation`` parameter:
            ``"scf"``, ``"nscf"``, ``"bands"``, ``"relax"``,
            ``"md"``, ``"vc-relax"``, ``"vc-md"``.
            Other codes may use different task identifiers.
        xc: Exchange-correlation functional.
            Supported: ``"pbesol"``.
            Planned: ``"pbe"``, ``"hse06"``, ``"scan"``, ``"r2scan"``.
        pseudo_family: Pseudopotential family label (aiida-pseudo format).
            Supported: ``"PseudoDojo/0.4/PBEsol/SR/standard/upf"``,
            ``"PseudoDojo/0.4/PBEsol/FR/standard/upf"``,
            ``"SSSP/1.3/PBEsol/efficiency"``,
            ``"SSSP/1.3/PBEsol/accurate"``.
            Planned: ``"SSSP/1.3/PBE/efficiency"``,
            ``"SSSP/1.3/PBE/accurate"``.
        accuracy: Calculation accuracy tier.
        hints: User overrides for any parameter (provenance = ``"user_hint"``).
    """

    structure: Structure
    code: str = "qe"
    task: Literal["scf", "nscf", "bands", "relax", "md", "vc-relax", "vc-md", "ph"] = "scf"
    xc: str = "pbesol"
    pseudo_family: str = "PseudoDojo/0.4/PBEsol/SR/standard/upf"
    accuracy: Literal["fast", "balanced", "accurate"] = "accurate"
    hints: dict[str, Any] = field(default_factory=dict)
