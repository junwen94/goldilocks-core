"""Assemble AiiDA PwBaseWorkChain / PwRelaxWorkChain inputs.

Requires: ``pip install goldilocks-core[aiida]``
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from goldilocks_core.advise.types import QEParameterSet
    from goldilocks_core.intent import CalculationIntent


def _require_aiida() -> None:
    try:
        import aiida  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "aiida-core is required for this function. "
            "Install it with: pip install goldilocks-core[aiida]"
        ) from exc


def build_pw_inputs(
    params: "QEParameterSet",
    intent: "CalculationIntent",
    code_label: str,
    pseudo_family: str | None = None,
    computer_metadata: dict[str, Any] | None = None,
) -> Any:
    """Build inputs dict for ``PwBaseWorkChain`` (or ``PwRelaxWorkChain``).

    Args:
        params: QEParameterSet from the advise pipeline.
        intent: Calculation intent (task, structure, etc.).
        code_label: AiiDA code label for pw.x (e.g. ``"pw-7.2@archer2"``).
        pseudo_family: aiida-pseudo family label.  When None, pseudos are
            loaded from local UPF files referenced in *params*.
        computer_metadata: Scheduler metadata (``{"num_machines": 1, ...}``).

    Returns:
        Dict suitable for ``PwBaseWorkChain.get_builder().update(...)``.
    """
    _require_aiida()

    from aiida import orm
    from aiida.plugins import WorkflowFactory

    from goldilocks_core.aiida.convert import kpoints_to_mesh_dict, qe_params_to_input_dict
    from goldilocks_core.aiida.pseudo import load_pseudos

    PwBaseWorkChain = WorkflowFactory("quantumespresso.pw.base")  # noqa: N806

    builder = PwBaseWorkChain.get_builder()
    builder.pw.code = orm.load_code(code_label)
    builder.pw.structure = orm.StructureData(pymatgen=intent.structure)

    # Parameters
    input_dict = qe_params_to_input_dict(params, intent)
    builder.pw.parameters = orm.Dict(dict=input_dict)

    # K-points
    kpoints = orm.KpointsData()
    mesh_dict = kpoints_to_mesh_dict(params)
    kpoints.set_kpoints_mesh(mesh_dict["mesh"], offset=mesh_dict["offset"])
    builder.kpoints = kpoints

    # Pseudopotentials
    builder.pw.pseudos = load_pseudos(params, pseudo_family=pseudo_family)

    # Scheduler metadata
    meta = computer_metadata or {"num_machines": 1, "num_mpiprocs_per_machine": 1}
    builder.pw.metadata.options.update(meta)

    return builder


def build_relax_inputs(
    params: "QEParameterSet",
    intent: "CalculationIntent",
    code_label: str,
    pseudo_family: str | None = None,
    relax_type: str = "positions_cell",
    computer_metadata: dict[str, Any] | None = None,
) -> Any:
    """Build inputs for ``PwRelaxWorkChain``.

    Args:
        relax_type: AiiDA relax type string — ``"positions"`` (relax),
            ``"positions_cell"`` (vc-relax), etc.
    """
    _require_aiida()

    from aiida import orm
    from aiida.plugins import WorkflowFactory

    from goldilocks_core.aiida.convert import kpoints_to_mesh_dict, qe_params_to_input_dict
    from goldilocks_core.aiida.pseudo import load_pseudos

    PwRelaxWorkChain = WorkflowFactory("quantumespresso.pw.relax")  # noqa: N806

    builder = PwRelaxWorkChain.get_builder()
    builder.structure = orm.StructureData(pymatgen=intent.structure)
    builder.base.pw.code = orm.load_code(code_label)

    input_dict = qe_params_to_input_dict(params, intent)
    builder.base.pw.parameters = orm.Dict(dict=input_dict)

    kpoints = orm.KpointsData()
    mesh_dict = kpoints_to_mesh_dict(params)
    kpoints.set_kpoints_mesh(mesh_dict["mesh"], offset=mesh_dict["offset"])
    builder.base.kpoints = kpoints

    builder.base.pw.pseudos = load_pseudos(params, pseudo_family=pseudo_family)
    builder.relax_type = orm.Str(relax_type)

    meta = computer_metadata or {"num_machines": 1, "num_mpiprocs_per_machine": 1}
    builder.base.pw.metadata.options.update(meta)

    return builder
