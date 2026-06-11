"""Generate Quantum ESPRESSO pw.x input files from a QEParameterSet.

Usage::

    from goldilocks_core.generate.qe import write_qe_inputs

    result = write_qe_inputs(params, structure, intent, output_dir="./goldilocks_output")
    # result["input_file"]  → Path to goldilocks.in
    # result["pseudo_dir"]  → Path to pseudo/ directory (pp files copied here)
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from pymatgen.core import Structure
from pymatgen.io.pwscf import PWInput

if TYPE_CHECKING:
    from goldilocks_core.advise.types import QEParameterSet
    from goldilocks_core.intent import CalculationIntent

_QE_TASK: dict[str, str] = {
    "scf":      "scf",
    "relax":    "relax",
    "vc-relax": "vc-relax",
    "vc_relax": "vc-relax",
    "nscf":     "nscf",
    "bands":    "bands",
    "md":       "md",
    "vc-md":    "vc-md",
}


def _species_order(structure: Structure) -> list[str]:
    """Return unique element symbols in first-appearance order (matches PWInput ATOMIC_SPECIES)."""
    seen: dict[str, None] = {}
    for site in structure:
        seen[str(site.specie)] = None
    return list(seen)


def _z_valence_map(params: "QEParameterSet") -> dict[str, float]:
    """Read valence electron counts from bundled UPF files."""
    from goldilocks_core.pseudo.parse_upf import parse_upf_metadata

    z_map: dict[str, float] = {}
    for ps in params.pseudos:
        if ps.path is not None and ps.path.exists():
            try:
                meta = parse_upf_metadata(ps.path)
                if meta.z_valence is not None:
                    z_map[ps.element] = meta.z_valence
            except Exception:
                pass
    return z_map


def write_qe_inputs(
    params: "QEParameterSet",
    structure: Structure,
    intent: "CalculationIntent",
    output_dir: str | Path = "./goldilocks_output",
) -> dict[str, Path]:
    """Generate ``goldilocks.in`` and copy pseudopotential files.

    Args:
        params: QEParameterSet from the advise pipeline.
        structure: Crystal structure (pymatgen).
        intent: CalculationIntent carrying task, accuracy, etc.
        output_dir: Directory to write files into (created if absent).

    Returns:
        ``{"input_file": Path, "pseudo_dir": Path}``

    The pseudopotential files referenced in *params* are copied to
    ``<output_dir>/pseudo/``.  The input file is written to
    ``<output_dir>/goldilocks.in``.
    """
    output_dir = Path(output_dir)
    pseudo_dir = output_dir / "pseudo"
    pseudo_dir.mkdir(parents=True, exist_ok=True)

    # ── Pseudopotential dict and file copy ──────────────────────────────────
    pseudo_dict: dict[str, str] = {}
    missing: list[str] = []
    for ps in params.pseudos:
        pseudo_dict[ps.element] = ps.filename
        if ps.path is not None and ps.path.exists():
            shutil.copy2(ps.path, pseudo_dir / ps.filename)
        else:
            missing.append(ps.element)

    # ── SYSTEM namelist ─────────────────────────────────────────────────────
    system: dict = {
        "ecutwfc":    params.ecutwfc,
        "ecutrho":    params.ecutrho,
        "occupations": params.occupations,
    }
    if params.smearing is not None:
        system["smearing"] = params.smearing
        system["degauss"]  = round(params.degauss, 6)

    # Spin flags — avoid setting nspin=4 alongside noncolin (QE convention)
    if params.noncolin:
        system["noncolin"] = True
        if params.lspinorb:
            system["lspinorb"] = True
    elif params.nspin == 2:
        system["nspin"] = 2

    # starting_magnetization: μB → QE fraction (divide by pseudo z_valence)
    if params.starting_magnetization:
        z_map = _z_valence_map(params)
        species = _species_order(structure)
        el_to_idx = {el: i + 1 for i, el in enumerate(species)}
        for el, mag_ub in params.starting_magnetization.items():
            idx = el_to_idx.get(el)
            if idx is None:
                continue
            z_val = z_map.get(el)
            if z_val and z_val > 0:
                mag_frac = max(-1.0, min(1.0, mag_ub / z_val))
            else:
                # Fallback: clamp μB to [-1, 1] directly
                mag_frac = max(-1.0, min(1.0, mag_ub / 10.0))
            system[f"starting_magnetization({idx})"] = round(mag_frac, 4)

        # angle1 / angle2 for non-collinear (degrees; FM default 0.0)
        for angle_dict, key_prefix in [
            (params.angle1, "angle1"),
            (params.angle2, "angle2"),
        ]:
            if angle_dict:
                for el, angle in angle_dict.items():
                    idx = el_to_idx.get(el)
                    if idx is not None:
                        system[f"{key_prefix}({idx})"] = angle

    # ── CONTROL namelist ────────────────────────────────────────────────────
    control: dict = {
        "calculation": _QE_TASK.get(intent.task, intent.task),
        "pseudo_dir":  "./pseudo",
        "outdir":      "./out",
        "prefix":      "goldilocks",
    }

    # ── Assemble and write ──────────────────────────────────────────────────
    pw = PWInput(
        structure=structure,
        pseudo=pseudo_dict,
        control=control,
        system=system,
        electrons={},
        ions={},
        cell={},
        kpoints_mode="automatic",
        kpoints_grid=params.kpoints_grid,
        kpoints_shift=params.kpoints_shift,
    )

    input_file = output_dir / "goldilocks.in"
    pw.write_file(str(input_file))

    return {"input_file": input_file, "pseudo_dir": pseudo_dir, "missing_pp": missing}
