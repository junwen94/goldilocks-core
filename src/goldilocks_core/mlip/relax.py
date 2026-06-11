"""Geometry relaxation using a MACE potential.

Requires: ``pip install goldilocks-core[mlip]``
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pymatgen.core import Structure

_FMAX_DEFAULT = 0.05        # eV/Å
_MAX_STEPS_DEFAULT = 500


def relax_structure(
    structure: "Structure",
    model: str | Path = "medium",
    fmax: float = _FMAX_DEFAULT,
    max_steps: int = _MAX_STEPS_DEFAULT,
    device: str = "cpu",
) -> tuple["Structure", float]:
    """Relax a structure with a MACE-MP potential using ASE BFGS.

    Args:
        structure: Input crystal structure (pymatgen).
        model: MACE-MP model size string (``"small"``, ``"medium"``,
            ``"large"``), or a local path to a ``.model`` checkpoint.
        fmax: Maximum residual force in eV/Å at convergence.
        max_steps: Maximum BFGS steps.
        device: ``"cpu"`` or ``"cuda"``.

    Returns:
        ``(relaxed_structure, final_energy_ev)``

    Raises:
        ImportError: if mace-torch is not installed.
        RuntimeError: if relaxation did not converge within *max_steps*.
    """
    try:
        from mace.calculators import mace_mp
    except ImportError as exc:
        raise ImportError(
            "mace-torch is required for MLIP relaxation. "
            "Install with: pip install goldilocks-core[mlip]"
        ) from exc

    from ase.optimize import BFGS
    from pymatgen.io.ase import AseAtomsAdaptor

    atoms = AseAtomsAdaptor.get_atoms(structure)
    calc = mace_mp(model=str(model), device=device, default_dtype="float64")
    atoms.calc = calc

    opt = BFGS(atoms, logfile=None)
    converged = opt.run(fmax=fmax, steps=max_steps)

    if not converged:
        raise RuntimeError(
            f"MACE relaxation did not converge within {max_steps} steps "
            f"(final fmax={opt.fmax:.4f} eV/Å, target={fmax})"
        )

    relaxed: "Structure" = AseAtomsAdaptor.get_structure(atoms)
    energy_ev: float = float(atoms.get_potential_energy())
    return relaxed, energy_ev
