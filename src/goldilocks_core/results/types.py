"""Result types for the Results Lab stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import numpy as np
    from pymatgen.core import Structure


@dataclass(frozen=True, slots=True)
class SCFResult:
    converged: bool
    energy_ev: float
    fermi_energy_ev: float | None
    total_magnetization: float | None  # Bohr mag/cell; None for non-magnetic runs
    n_iterations: int
    warnings: list[str]


@dataclass(frozen=True, slots=True)
class RelaxResult:
    converged: bool
    final_structure: "Structure | None"  # None when structure could not be parsed
    final_energy_ev: float
    n_ionic_steps: int
    warnings: list[str]


@dataclass(frozen=True, slots=True)
class BandResult:
    eigenvalues_ev: "np.ndarray"    # shape (nkpts, nbands)
    kpoints: "np.ndarray"           # shape (nkpts, 3), crystal coordinates
    fermi_energy_ev: float
    warnings: list[str]


@dataclass(frozen=True, slots=True)
class ValidationWarning:
    level: Literal["error", "warning", "info"]
    parameter: str
    message: str
    suggestion: str | None = None


@dataclass(frozen=True, slots=True)
class ValidationReport:
    passed: list[str]
    warnings: list[ValidationWarning]
    manifest: dict | None  # raw manifest dict; None when manifest.json not found
