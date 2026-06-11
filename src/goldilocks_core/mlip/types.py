"""Data types for the MLIP pre-processing stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pymatgen.core import Structure


@dataclass(frozen=True, slots=True)
class MLIPPreview:
    """Summary of MLIP-based pre-analysis results.

    Fields that could not be computed (task not requested or failed)
    are None.  The dataclass itself has no dependency on mace-torch;
    import it freely even when the optional extra is not installed.
    """

    # Geometry (relax.py, v1)
    relaxed_structure: "Structure | None"
    final_energy_ev: float | None

    # Phonon stability (phonon.py, v1)
    phonon_stable: bool | None
    imaginary_frequencies: list[float] = field(default_factory=list)

    # EOS (eos.py, v2 — not yet implemented)
    equilibrium_volume_a3: float | None = None
    bulk_modulus_gpa: float | None = None

    # Metadata
    model_used: str = ""
    tasks_run: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
