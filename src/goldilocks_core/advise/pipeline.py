from __future__ import annotations

from goldilocks_core.advise.basis import advise_basis
from goldilocks_core.advise.kpoints import advise_kpoints
from goldilocks_core.advise.protocol import select_protocol
from goldilocks_core.advise.pseudo import advise_pseudos
from goldilocks_core.advise.smearing import advise_smearing
from goldilocks_core.advise.spin import advise_spin
from goldilocks_core.advise.types import AdviceBundle
from goldilocks_core.advise.vdw import advise_vdw
from goldilocks_core.analyse.structure import StructureAnalysis
from goldilocks_core.intent import CalculationIntent


def advise(
    analysis: StructureAnalysis,
    intent: CalculationIntent,
    k_index: int | None = None,
    k_distance_ml: float | None = None,
) -> AdviceBundle:
    """Run the full heuristic advise pipeline and return a code-agnostic AdviceBundle.

    Args:
        k_index: ML-predicted k_index (k_index metric). None → heuristic.
        k_distance_ml: ML-predicted k_distance (Å⁻¹, k_distance metric).
            k_index takes priority if both are provided.

    Returns:
        AdviceBundle with code-agnostic decisions and provenance for every
        parameter.  Pass to ``goldilocks_core.select.qe.build_qe_parameter_set``
        to obtain QE-specific concrete values.
    """
    protocol, _ = select_protocol(analysis, intent)

    smearing_decision = advise_smearing(analysis, intent, protocol)
    kpoints_decision  = advise_kpoints(analysis, intent, protocol, k_index, k_distance_ml)
    spin_decision     = advise_spin(analysis, intent)
    pseudos           = advise_pseudos(analysis, intent)
    cutoff_decision   = advise_basis(analysis, intent, pseudos)
    vdw_decision      = advise_vdw(analysis, intent)

    return AdviceBundle(
        smearing=smearing_decision,
        kpoints=kpoints_decision,
        spin=spin_decision,
        pseudos=pseudos,
        cutoff=cutoff_decision,
        vdw=vdw_decision,
    )
