"""High-level orchestrator for MLIP pre-analysis.

Requires: ``pip install goldilocks-core[mlip]``
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pymatgen.core import Structure

    from goldilocks_core.mlip.types import MLIPPreview

_Task = Literal["relax", "phonon"]


def run_mlip_prep(
    structure: "Structure",
    tasks: list[_Task] | None = None,
    model: str | Path = "medium",
    device: str = "cpu",
    fmax: float = 0.05,
    relax_max_steps: int = 500,
    phonon_supercell: tuple[int, int, int] = (2, 2, 2),
) -> "MLIPPreview":
    """Run MLIP pre-analysis tasks and return a summary preview.

    Args:
        structure: Input crystal structure (pymatgen).
        tasks: List of tasks to run — ``["relax"]``, ``["phonon"]``, or
            ``["relax", "phonon"]``.  Defaults to ``["relax"]``.
        model: MACE-MP model size (``"small"``, ``"medium"``, ``"large"``)
            or a local checkpoint path.
        device: ``"cpu"`` or ``"cuda"``.
        fmax: Convergence threshold for geometry relaxation (eV/Å).
        relax_max_steps: Maximum BFGS steps for relaxation.
        phonon_supercell: Supercell dimensions for phonon calculation.

    Returns:
        MLIPPreview summarising all completed tasks.

    Raises:
        ImportError: if mace-torch is not installed.
    """
    from goldilocks_core.mlip.types import MLIPPreview

    tasks = tasks or ["relax"]
    warnings: list[str] = []
    tasks_run: list[str] = []

    relaxed_structure: "Structure | None" = None
    final_energy_ev: float | None = None
    phonon_stable: bool | None = None
    imaginary_frequencies: list[float] = []

    # ── Geometry relaxation ──────────────────────────────────────────────────
    if "relax" in tasks:
        try:
            from goldilocks_core.mlip.relax import relax_structure

            relaxed_structure, final_energy_ev = relax_structure(
                structure,
                model=model,
                fmax=fmax,
                max_steps=relax_max_steps,
                device=device,
            )
            tasks_run.append("relax")
        except RuntimeError as exc:
            warnings.append(f"Relaxation did not converge: {exc}")
            tasks_run.append("relax(failed)")

    # ── Phonon stability ─────────────────────────────────────────────────────
    if "phonon" in tasks:
        source = relaxed_structure if relaxed_structure is not None else structure
        try:
            from goldilocks_core.mlip.phonon import check_phonon_stability

            phonon_stable, imaginary_frequencies = check_phonon_stability(
                source,
                model=str(model),
                supercell=phonon_supercell,
                device=device,
            )
            tasks_run.append("phonon")
            if not phonon_stable:
                n = len(imaginary_frequencies)
                warnings.append(
                    f"Phonon instability detected: {n} imaginary mode(s), "
                    f"min = {min(imaginary_frequencies):.2f} THz. "
                    "Verify DFT relaxation before production run."
                )
        except ImportError as exc:
            warnings.append(f"Phonon check skipped — {exc}")
        except Exception as exc:
            warnings.append(f"Phonon check failed: {exc}")

    try:
        from mace.calculators import mace_mp  # noqa: F401
        model_label = f"mace-mp-{model}" if isinstance(model, str) else Path(model).stem
    except ImportError:
        model_label = "mace-mp (unknown)"

    return MLIPPreview(
        relaxed_structure=relaxed_structure,
        final_energy_ev=final_energy_ev,
        phonon_stable=phonon_stable,
        imaginary_frequencies=imaginary_frequencies,
        model_used=model_label,
        tasks_run=tasks_run,
        warnings=warnings,
    )
