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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pymatgen.core import Structure

    from goldilocks_core.advise.types import QEParameterSet
    from goldilocks_core.analyse.structure import StructureAnalysis
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


def _hints_to_dict(hints: object) -> dict:
    """Serialise ParameterHints to a plain dict, omitting None values."""
    import dataclasses
    if dataclasses.is_dataclass(hints) and not isinstance(hints, type):
        return {k: v for k, v in dataclasses.asdict(hints).items() if v is not None}
    if isinstance(hints, dict):
        return {k: v for k, v in hints.items() if v is not None}
    return {}


def _build_manifest(
    params: "QEParameterSet",
    structure: "Structure",
    intent: "CalculationIntent",
    analysis: "StructureAnalysis | None",
) -> dict:
    """Build a JSON-serialisable manifest dict for this run."""
    from datetime import datetime, timezone

    try:
        from importlib.metadata import version
        gl_version = version("goldilocks-core")
    except Exception:
        gl_version = "unknown"

    doc: dict = {
        "goldilocks_version": gl_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "structure": {
            "formula": structure.composition.reduced_formula,
            "n_atoms": len(structure),
            "n_species": len(structure.composition),
        },
        "intent": {
            "task": intent.task,
            "code": getattr(intent, "code", "qe"),
            "xc": getattr(intent, "xc", "pbesol"),
            "pseudo_family": intent.pseudo_family,
            "accuracy": intent.accuracy,
            "hints": _hints_to_dict(intent.hints),
        },
        "parameters": {
            "spin": {
                "treatment": params.spin_decision.treatment,
                "provenance": params.spin_decision.provenance,
            },
            "smearing": {
                "use_smearing": params.smearing_decision.use_smearing,
                "method": params.smearing_decision.method,
                "width_ev": params.smearing_decision.width_ev,
                "provenance": params.smearing_decision.provenance,
            },
            "kpoints": {
                "grid": list(params.kpoints_grid),
                "shift": list(params.kpoints_shift),
                "provenance": params.kpoints_decision.provenance,
            },
            "cutoffs": {
                "ecutwfc_ry": params.ecutwfc,
                "ecutrho_ry": params.ecutrho,
                "provenance": params.cutoff_decision.provenance,
            },
            "vdw": {
                "use_vdw": params.vdw_decision.use_vdw,
                "method": params.vdw_decision.method,
                "provenance": params.vdw_decision.provenance,
            },
            "pseudos": [
                {
                    "element": ps.element,
                    "family": ps.family,
                    "filename": ps.filename,
                    "provenance": ps.provenance,
                }
                for ps in params.pseudos
            ],
        },
    }

    if analysis is not None:
        doc["analysis"] = {
            "metallicity": analysis.metallicity,
            "metallicity_source": analysis.metallicity_source,
            "magnetic_elements": list(analysis.magnetic_elements),
            "soc_relevant": analysis.soc_relevant,
            "heavy_elements": list(analysis.heavy_elements),
            "dimensionality": analysis.dimensionality,
            "system_type": analysis.system_type,
            "has_vacuum": analysis.has_vacuum,
            "warnings": list(analysis.warnings),
        }

    return doc


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
    structure: "Structure",
    intent: "CalculationIntent",
    output_dir: str | Path = "./goldilocks_output",
    kgrid_override: tuple[int, int, int] | None = None,
    conv_thr: float | None = None,
    analysis: "StructureAnalysis | None" = None,
) -> dict[str, Any]:
    """Generate ``goldilocks.in`` and copy pseudopotential files.

    Args:
        params: QEParameterSet from the advise pipeline.
        structure: Crystal structure (pymatgen).
        intent: CalculationIntent carrying task, accuracy, etc.
        output_dir: Directory to write files into (created if absent).
        kgrid_override: Replace the k-grid from *params* (e.g. commensurate
            phonon k-grid from ``PhononSetupAdvice.phonon_kgrid``).
        conv_thr: Override ``&ELECTRONS conv_thr`` (e.g. 1e-10 for phonon SCF).

    Returns:
        ``{"input_file": Path, "pseudo_dir": Path, "missing_pp": list[str]}``

    The pseudopotential files referenced in *params* are copied to
    ``<output_dir>/pseudo/``.  The input file is written to
    ``<output_dir>/goldilocks.in``.
    """
    from ase.io.espresso import write_espresso_in
    from pymatgen.io.ase import AseAtomsAdaptor

    output_dir = Path(output_dir)
    pseudo_dir = output_dir / "pseudo"
    pseudo_dir.mkdir(parents=True, exist_ok=True)

    # ── Structure conversion ────────────────────────────────────────────────
    atoms = AseAtomsAdaptor.get_atoms(structure)

    # ── Pseudopotential dict and file copy ──────────────────────────────────
    pseudopotentials: dict[str, str] = {}
    missing: list[str] = []
    for ps in params.pseudos:
        pseudopotentials[ps.element] = ps.filename
        if ps.path is not None and ps.path.exists():
            shutil.copy2(ps.path, pseudo_dir / ps.filename)
        else:
            missing.append(ps.element)

    # ── SYSTEM namelist ─────────────────────────────────────────────────────
    system: dict = {
        "ecutwfc":     round(params.ecutwfc, 6),
        "ecutrho":     round(params.ecutrho, 6),
        "occupations": params.occupations,
    }
    if params.smearing is not None:
        system["smearing"] = params.smearing
        system["degauss"]  = round(params.degauss, 6)

    if params.vdw_corr is not None:
        system["vdw_corr"] = params.vdw_corr

    # Spin flags — avoid setting nspin=4 alongside noncolin (QE convention)
    if params.noncolin:
        system["noncolin"] = True
        if params.lspinorb:
            system["lspinorb"] = True
    elif params.nspin == 2:
        system["nspin"] = 2

    # starting_magnetization: μB → QE fraction (divide by pseudo z_valence)
    # ASE reads initial magmoms from the Atoms object and writes
    # starting_magnetization(N) from there — so we set them on the atoms
    # object instead of injecting into input_data["system"].
    if params.starting_magnetization:
        z_map = _z_valence_map(params)
        seen: dict[str, None] = {}
        for sym in atoms.get_chemical_symbols():
            seen[sym] = None
        el_to_idx = {el: i + 1 for i, el in enumerate(seen)}

        # Compute per-element fractional magmoms
        mag_fracs: dict[str, float] = {}
        for el, mag_ub in params.starting_magnetization.items():
            z_val = z_map.get(el)
            if z_val and z_val > 0:
                mag_fracs[el] = round(max(-1.0, min(1.0, mag_ub / z_val)), 4)
            else:
                mag_fracs[el] = round(max(-1.0, min(1.0, mag_ub / 10.0)), 4)

        # Set per-atom initial magnetic moments so ASE writes correct values
        atoms.set_initial_magnetic_moments(
            [mag_fracs.get(sym, 0.0) for sym in atoms.get_chemical_symbols()]
        )

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

    # ── Assemble input_data (nested namelists) ──────────────────────────────
    electrons: dict = {}
    if conv_thr is not None:
        electrons["conv_thr"] = conv_thr

    input_data = {
        "control": {
            "calculation": _QE_TASK.get(intent.task, intent.task),
            "pseudo_dir":  "./pseudo",
            "outdir":      "./out",
            "prefix":      "goldilocks",
        },
        "system":    system,
        "electrons": electrons,
    }

    kgrid  = kgrid_override if kgrid_override is not None else params.kpoints_grid

    # ── Write input file ────────────────────────────────────────────────────
    input_file = output_dir / f"gl-pw-{intent.task}.in"
    with open(input_file, "w") as fh:
        write_espresso_in(
            fh,
            atoms,
            input_data=input_data,
            pseudopotentials=pseudopotentials,
            kpts=kgrid,
            koffset=tuple(params.kpoints_shift),
        )

    import json
    manifest = _build_manifest(params, structure, intent, analysis)
    manifest_file = output_dir / "goldilocks_manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    return {
        "input_file": input_file,
        "pseudo_dir": pseudo_dir,
        "missing_pp": missing,
        "manifest_file": manifest_file,
    }


def write_ph_inputs(
    output_dir: str | Path = "./goldilocks_output",
    nq: tuple[int, int, int] | None = None,
    qpts: tuple[float, float, float] = (0.0, 0.0, 0.0),
    epsil: bool = False,
    tr2_ph: float = 1.0e-14,
) -> dict[str, Path]:
    """Generate a ``ph.in`` input file for QE ph.x.

    Args:
        output_dir: Directory to write ``ph.in`` into.
        nq: q-point grid ``(nq1, nq2, nq3)`` for a full phonon dispersion.
            When provided, ``ldisp = True`` is set and *qpts* is ignored.
        qpts: Single q-point in units of ``2π/a`` for a Gamma-point (or
            specific q-point) calculation.  Default: Gamma ``(0, 0, 0)``.
        epsil: If True, compute Born effective charges and dielectric tensor
            (required for LO-TO splitting in polar insulators).
        tr2_ph: Phonon self-consistency threshold (default 1e-14).

    Returns:
        ``{"ph_file": Path}``

    The file is written to ``<output_dir>/ph.in``.  A matching pw.x SCF
    calculation (same ``prefix`` / ``outdir``) must be completed first.
    """
    from ase.io.espresso import write_espresso_ph

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inputph: dict = {
        "prefix": "goldilocks",
        "outdir": "./out",
        "fildyn": "goldilocks.dyn",
        "tr2_ph": tr2_ph,
    }
    if epsil:
        inputph["epsil"] = True

    if nq is not None:
        inputph["ldisp"] = True
        inputph["nq1"]   = nq[0]
        inputph["nq2"]   = nq[1]
        inputph["nq3"]   = nq[2]
        ph_qpts: tuple = (0.0, 0.0, 0.0)  # ignored when ldisp=True
    else:
        ph_qpts = qpts

    ph_file = output_dir / "gl-ph.in"
    with open(ph_file, "w") as fh:
        write_espresso_ph(fh, input_data={"inputph": inputph}, qpts=ph_qpts)

    return {"ph_file": ph_file}
