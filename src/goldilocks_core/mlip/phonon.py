"""Phonon stability check using finite-difference force constants (MACE).

Requires: ``pip install goldilocks-core[mlip]``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pymatgen.core import Structure

_IMAGINARY_THRESHOLD_THZ = -0.1   # frequencies below this (in THz) are imaginary
_SUPERCELL_DEFAULT = (2, 2, 2)


def check_phonon_stability(
    structure: "Structure",
    model: str = "medium",
    supercell: tuple[int, int, int] = _SUPERCELL_DEFAULT,
    displacement: float = 0.01,
    device: str = "cpu",
) -> tuple[bool, list[float]]:
    """Estimate phonon stability via finite-difference force constants.

    Uses phonopy to build displacements and MACE to compute forces.

    Args:
        structure: Crystal structure (pymatgen).
        model: MACE-MP model size (``"small"``, ``"medium"``, ``"large"``)
            or local path to a checkpoint.
        supercell: Supercell matrix diagonal ``(n1, n2, n3)``.
        displacement: Finite-difference displacement in Å.
        device: ``"cpu"`` or ``"cuda"``.

    Returns:
        ``(stable, imaginary_frequencies)`` where *stable* is True when no
        imaginary modes are found, and *imaginary_frequencies* lists all
        frequencies (in THz) that are below the threshold.

    Raises:
        ImportError: if mace-torch or phonopy is not installed.
    """
    try:
        from mace.calculators import mace_mp
    except ImportError as exc:
        raise ImportError(
            "mace-torch is required. Install with: pip install goldilocks-core[mlip]"
        ) from exc
    try:
        import phonopy  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "phonopy is required for phonon stability checks. "
            "Install with: pip install phonopy"
        ) from exc

    import numpy as np
    import phonopy
    from phonopy.interface.pymatgen import get_phonopy_structure
    from pymatgen.io.ase import AseAtomsAdaptor

    # Build phonopy object
    pm_structure = get_phonopy_structure(structure)
    ph = phonopy.Phonopy(
        pm_structure,
        supercell_matrix=np.diag(supercell),
        factor=phonopy.units.VaspToTHz,
    )
    ph.generate_displacements(distance=displacement)

    # Compute forces for each displaced supercell with MACE
    calc = mace_mp(model=str(model), device=device, default_dtype="float64")
    forces: list[np.ndarray] = []
    for sc in ph.supercells_with_displacements:
        atoms = AseAtomsAdaptor.get_atoms(
            _phonopy_to_pymatgen(sc)
        )
        atoms.calc = calc
        forces.append(atoms.get_forces())

    ph.forces = np.array(forces)
    ph.produce_force_constants()
    ph.symmetrize_force_constants()

    # Sample the Brillouin zone on a Gamma-centred mesh
    ph.run_mesh(mesh=[20, 20, 20])
    ph.run_total_dos()

    mesh = ph.get_mesh_dict()
    freqs: np.ndarray = mesh["frequencies"].ravel()

    imaginary = [float(f) for f in freqs if f < _IMAGINARY_THRESHOLD_THZ]
    return len(imaginary) == 0, sorted(imaginary)


def _phonopy_to_pymatgen(phonopy_atoms: "object") -> "Structure":  # type: ignore[misc]
    """Convert a phonopy Atoms object to a pymatgen Structure."""
    from pymatgen.core import Lattice, Structure

    pa = phonopy_atoms  # type: ignore[assignment]
    return Structure(
        lattice=Lattice(pa.cell),  # type: ignore[attr-defined]
        species=pa.symbols,  # type: ignore[attr-defined]
        coords=pa.scaled_positions,  # type: ignore[attr-defined]
    )
