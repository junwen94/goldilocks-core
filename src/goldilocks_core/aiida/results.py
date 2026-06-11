"""Load AiiDA calculation results into goldilocks result types.

Requires: ``pip install goldilocks-core[aiida]``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from goldilocks_core.results.types import RelaxResult, SCFResult

_EV_PER_RY: float = 13.605693122994


def load_scf_result(pk: int) -> "SCFResult":
    """Load SCF results from a finished ``PwBaseWorkChain`` node.

    Args:
        pk: Primary key of the completed workchain.

    Returns:
        SCFResult populated from the workchain outputs.
    """
    try:
        from aiida import orm
    except ImportError as exc:
        raise ImportError(
            "aiida-core is required. Install with: pip install goldilocks-core[aiida]"
        ) from exc

    from goldilocks_core.results.types import SCFResult

    node = orm.load_node(pk)
    outputs = node.outputs

    energy_ry: float = float(outputs.output_parameters["energy"])
    energy_ev = energy_ry * _EV_PER_RY

    fermi_ev: float | None = outputs.output_parameters.get_dict().get("fermi_energy")
    total_mag: float | None = outputs.output_parameters.get_dict().get("total_magnetization")
    n_iter: int = int(outputs.output_parameters.get_dict().get("scf_iterations", 0))
    warnings: list[str] = outputs.output_parameters.get_dict().get("warnings", [])
    converged: bool = not outputs.output_parameters.get_dict().get("convergence_info", {}).get(
        "electron_convergence_failed", False
    )

    return SCFResult(
        converged=converged,
        energy_ev=energy_ev,
        fermi_energy_ev=fermi_ev,
        total_magnetization=total_mag,
        n_iterations=n_iter,
        warnings=warnings if isinstance(warnings, list) else [str(warnings)],
    )


def load_relax_result(pk: int) -> "RelaxResult":
    """Load relax results from a finished ``PwRelaxWorkChain`` node.

    Args:
        pk: Primary key of the completed workchain.

    Returns:
        RelaxResult with final structure and energy.
    """
    try:
        from aiida import orm
    except ImportError as exc:
        raise ImportError(
            "aiida-core is required. Install with: pip install goldilocks-core[aiida]"
        ) from exc

    from goldilocks_core.results.types import RelaxResult

    node = orm.load_node(pk)
    outputs = node.outputs

    energy_ry: float = float(outputs.output_parameters["energy"])
    final_energy_ev = energy_ry * _EV_PER_RY

    final_structure = outputs.output_structure.get_pymatgen()
    n_steps: int = int(outputs.output_parameters.get_dict().get("number_of_ionic_steps", 0))
    warnings: list[str] = outputs.output_parameters.get_dict().get("warnings", [])
    converged: bool = outputs.output_parameters.get_dict().get("geometry_converged", True)

    return RelaxResult(
        converged=converged,
        final_structure=final_structure,
        final_energy_ev=final_energy_ev,
        n_ionic_steps=n_steps,
        warnings=warnings if isinstance(warnings, list) else [str(warnings)],
    )
