"""Convert goldilocks types to AiiDA input dicts.

This module is pure Python — it does NOT import aiida-core, so it can be
used as a building block even when the ``aiida`` optional extra is not installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from goldilocks_core.advise.types import QEParameterSet
    from goldilocks_core.intent import CalculationIntent


def qe_params_to_input_dict(
    params: "QEParameterSet",
    intent: "CalculationIntent",
) -> dict[str, Any]:
    """Build the nested ``parameters`` dict expected by AiiDA's ``PwBaseWorkChain``.

    The returned dict maps to ``orm.Dict`` in AiiDA.  Keys follow the QE
    namelist structure: ``{"CONTROL": {...}, "SYSTEM": {...}, "ELECTRONS": {...}}``.

    Args:
        params: QE parameter set from the advise/select pipeline.
        intent: Calculation intent (provides task, outdir prefix, etc.).

    Returns:
        Plain Python dict suitable for ``orm.Dict(dict=...)``.
    """
    _QE_TASK = {
        "scf": "scf", "relax": "relax", "vc-relax": "vc-relax",
        "vc_relax": "vc-relax", "nscf": "nscf", "bands": "bands",
        "md": "md", "vc-md": "vc-md",
    }

    control: dict[str, Any] = {
        "calculation": _QE_TASK.get(intent.task, intent.task),
        "pseudo_dir": "./pseudo",
        "outdir": "./out",
        "prefix": "goldilocks",
    }

    system: dict[str, Any] = {
        "ecutwfc": round(params.ecutwfc, 6),
        "ecutrho": round(params.ecutrho, 6),
        "occupations": params.occupations,
    }

    if params.smearing is not None:
        system["smearing"] = params.smearing
        system["degauss"] = round(params.degauss, 6)

    if params.vdw_corr is not None:
        system["vdw_corr"] = params.vdw_corr

    if params.noncolin:
        system["noncolin"] = True
        if params.lspinorb:
            system["lspinorb"] = True
    elif params.nspin == 2:
        system["nspin"] = 2

    if params.starting_magnetization:
        # QE uses species index (1-based) as the key: starting_magnetization(N)
        # Build element → index map from the ordered pseudo list
        el_to_idx = {ps.element: i + 1 for i, ps in enumerate(params.pseudos)}
        for el, mag_ub in params.starting_magnetization.items():
            idx = el_to_idx.get(el, 1)
            system[f"starting_magnetization({idx})"] = round(
                max(-1.0, min(1.0, mag_ub / 10.0)), 4
            )

    return {"CONTROL": control, "SYSTEM": system, "ELECTRONS": {}}


def kpoints_to_mesh_dict(params: "QEParameterSet") -> dict[str, Any]:
    """Build the k-points mesh dict expected by ``orm.KpointsData``.

    Returns:
        ``{"mesh": [nx, ny, nz], "offset": [sx, sy, sz]}``
    """
    return {
        "mesh": list(params.kpoints_grid),
        "offset": list(params.kpoints_shift),
    }


def pseudos_to_upf_dict(params: "QEParameterSet") -> dict[str, str]:
    """Map element symbols to pseudo filenames.

    The returned dict is used as ``pseudos`` in ``PwBaseWorkChain`` when
    loading ``UpfData`` nodes by filename, or as a reference for aiida-pseudo
    family lookups.
    """
    return {ps.element: ps.filename for ps in params.pseudos}
