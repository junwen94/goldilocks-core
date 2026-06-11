"""Parse Quantum ESPRESSO pw.x text output files.

Supports pw.x stdout files for SCF and relax/vc-relax calculations.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pymatgen.core import Structure

    from goldilocks_core.results.types import RelaxResult, SCFResult

_RY_TO_EV: float = 13.605693122994

_RE_CONVERGED = re.compile(
    r"convergence has been achieved in\s+(\d+)\s+iterations",
    re.IGNORECASE,
)
_RE_NOT_CONVERGED = re.compile(
    r"convergence NOT achieved|maximum number of steps",
    re.IGNORECASE,
)
_RE_ENERGY = re.compile(
    r"!\s+total energy\s+=\s+([-\d.]+)\s+Ry",
    re.IGNORECASE,
)
_RE_FERMI = re.compile(
    r"the Fermi energy is\s+([-\d.]+)\s+ev",
    re.IGNORECASE,
)
_RE_TOTAL_MAG = re.compile(
    r"total magnetization\s+=\s+([-\d.]+)\s+Bohr mag/cell",
    re.IGNORECASE,
)
_RE_WARNING = re.compile(r"%\s*Warning\s*:?\s*(.+)", re.IGNORECASE)
_RE_RELAX_DONE = re.compile(r"End of (?:BFGS )?Geometry Optimization", re.IGNORECASE)
_RE_BFGS_STEP = re.compile(r"BFGS step number\s*=?\s*(\d+)", re.IGNORECASE)
_RE_CELL_PARAMS = re.compile(
    r"CELL_PARAMETERS\s*\(?(?:angstrom)?\)?\s*\n((?:\s*[-\d.eE+]+\s+[-\d.eE+]+\s+[-\d.eE+]+\s*\n){3})",
    re.IGNORECASE,
)
_RE_ATOMIC_POSITIONS = re.compile(
    r"ATOMIC_POSITIONS\s*\(?(crystal|angstrom|bohr|alat)\)?\s*\n((?:\s*[A-Za-z]+\s+[-\d.eE+ ]+\n)+)",
    re.IGNORECASE,
)


def parse_scf(path: str | Path) -> "SCFResult":
    """Parse a pw.x SCF (or NSCF) stdout file.

    Args:
        path: Path to the pw.x output file.

    Returns:
        SCFResult populated from the file.
    """
    from goldilocks_core.results.types import SCFResult

    text = Path(path).read_text(errors="replace")

    m_conv = _RE_CONVERGED.search(text)
    converged = bool(m_conv) and not bool(_RE_NOT_CONVERGED.search(text))
    n_iterations = int(m_conv.group(1)) if m_conv else 0

    m_e = _RE_ENERGY.search(text)
    energy_ev = float(m_e.group(1)) * _RY_TO_EV if m_e else 0.0

    m_f = _RE_FERMI.search(text)
    fermi_ev = float(m_f.group(1)) if m_f else None

    m_mag = _RE_TOTAL_MAG.search(text)
    total_mag = float(m_mag.group(1)) if m_mag else None

    warnings = [m.group(1).strip() for m in _RE_WARNING.finditer(text)]

    return SCFResult(
        converged=converged,
        energy_ev=energy_ev,
        fermi_energy_ev=fermi_ev,
        total_magnetization=total_mag,
        n_iterations=n_iterations,
        warnings=warnings,
    )


def parse_relax(path: str | Path) -> "RelaxResult":
    """Parse a pw.x relax or vc-relax stdout file.

    Args:
        path: Path to the pw.x output file.

    Returns:
        RelaxResult with final structure, energy, and step count.
    """
    from goldilocks_core.results.types import RelaxResult

    text = Path(path).read_text(errors="replace")

    converged = bool(_RE_RELAX_DONE.search(text))

    # Final energy: last occurrence wins
    all_energies = _RE_ENERGY.findall(text)
    final_energy_ev = float(all_energies[-1]) * _RY_TO_EV if all_energies else 0.0

    # Number of ionic steps
    bfgs_steps = _RE_BFGS_STEP.findall(text)
    n_ionic_steps = max((int(s) for s in bfgs_steps), default=0)
    if n_ionic_steps == 0 and converged:
        n_ionic_steps = 1  # at least one step if converged

    warnings = [m.group(1).strip() for m in _RE_WARNING.finditer(text)]

    structure = _parse_final_structure(text)

    return RelaxResult(
        converged=converged,
        final_structure=structure,
        final_energy_ev=final_energy_ev,
        n_ionic_steps=n_ionic_steps,
        warnings=warnings,
    )


def _parse_final_structure(text: str) -> "Structure | None":
    """Extract the last ATOMIC_POSITIONS (and optionally CELL_PARAMETERS) block."""
    try:
        from pymatgen.core import Lattice, Structure
    except ImportError:
        return None

    pos_matches = list(_RE_ATOMIC_POSITIONS.finditer(text))
    if not pos_matches:
        return None

    last = pos_matches[-1]
    coord_type = last.group(1).lower()
    pos_block = last.group(2).strip()

    species: list[str] = []
    coords: list[list[float]] = []
    for line in pos_block.splitlines():
        parts = line.split()
        if len(parts) >= 4:
            species.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

    if not species:
        return None

    # Try to find the last CELL_PARAMETERS block (vc-relax only)
    cell_matches = list(_RE_CELL_PARAMS.finditer(text))
    if cell_matches:
        cell_block = cell_matches[-1].group(1).strip().splitlines()
        try:
            matrix = [[float(x) for x in row.split()] for row in cell_block if row.strip()]
            lattice = Lattice(matrix)
        except Exception:
            lattice = Lattice.cubic(1.0)
    else:
        lattice = Lattice.cubic(1.0)

    cart = coord_type == "angstrom"
    try:
        return Structure(lattice=lattice, species=species, coords=coords, coords_are_cartesian=cart)
    except Exception:
        return None
