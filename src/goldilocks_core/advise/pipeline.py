from __future__ import annotations

from goldilocks_core.advise.basis import advise_basis
from goldilocks_core.advise.kpoints import advise_kpoints
from goldilocks_core.advise.protocol import select_protocol
from goldilocks_core.advise.pseudo import advise_pseudos
from goldilocks_core.advise.smearing import advise_smearing
from goldilocks_core.advise.spin import advise_spin
from goldilocks_core.advise.types import QEParameterSet
from goldilocks_core.advise.vdw import advise_vdw
from goldilocks_core.analyse.structure import StructureAnalysis
from goldilocks_core.intent import CalculationIntent

_EV_TO_RY: float = 1 / 13.605693122994

_VDW_TO_QE: dict[str, str] = {
    "d3":   "grimme-d3",
    "d3bj": "grimme-d3bj",
    "ts":   "ts-vdw",
    "mbd":  "many-body-dispersion",
}

_SMEARING_TO_QE: dict[str, str] = {
    "marzari_vanderbilt": "mv",
    "methfessel_paxton":  "mp",
    "fermi_dirac":        "fd",
    "gaussian":           "gauss",
}

# (nspin, noncolin, lspinorb)
_SPIN_TO_QE: dict[str, tuple[int, bool, bool]] = {
    "non_magnetic":      (1, False, False),
    "collinear":         (2, False, False),
    "non_collinear":     (4, True,  False),
    "non_collinear_soc": (4, True,  True),
}


def build_qe_parameter_set(
    analysis: StructureAnalysis,
    intent: CalculationIntent,
    k_index: int | None = None,
    k_distance_ml: float | None = None,
) -> QEParameterSet:
    """Assemble a QEParameterSet from structure analysis and calculation intent.

    Runs the full heuristic advise pipeline in order:
      protocol → smearing → kpoints → spin → pseudos → basis

    Args:
        k_index: ML-predicted k_index (k_index metric). None → heuristic.
        k_distance_ml: ML-predicted k_distance (Å⁻¹, k_distance metric).
            k_index takes priority if both are provided.

    Returns:
        QEParameterSet with concrete QE parameters and code-agnostic
        decision provenance attached.
    """
    protocol, _ = select_protocol(analysis, intent)

    smearing_decision = advise_smearing(analysis, intent, protocol)
    kpoints_decision  = advise_kpoints(analysis, intent, protocol, k_index, k_distance_ml)
    spin_decision     = advise_spin(analysis, intent)
    pseudos           = advise_pseudos(analysis, intent)
    cutoff_decision   = advise_basis(analysis, intent, pseudos)
    vdw_decision      = advise_vdw(analysis, intent)

    # Smearing → QE SYSTEM card
    if smearing_decision.use_smearing:
        occupations: str = "smearing"
        smearing_qe: str | None = _SMEARING_TO_QE[smearing_decision.method]  # type: ignore[index]
        degauss = (smearing_decision.width_ev or 0.0) * _EV_TO_RY
    else:
        occupations = "fixed"
        smearing_qe = None
        degauss = 0.0

    # Spin → QE SYSTEM card
    nspin, noncolin, lspinorb = _SPIN_TO_QE[spin_decision.treatment]

    ecutwfc = cutoff_decision.wavefunction_cutoff_ev * _EV_TO_RY
    ecutrho = cutoff_decision.density_cutoff_ev * _EV_TO_RY

    vdw_corr = _VDW_TO_QE[vdw_decision.method] if vdw_decision.use_vdw and vdw_decision.method else None

    return QEParameterSet(
        occupations=occupations,      # type: ignore[arg-type]
        smearing=smearing_qe,
        degauss=degauss,
        kpoints_grid=kpoints_decision.grid,
        kpoints_shift=kpoints_decision.shift,
        pseudos=pseudos,
        ecutwfc=ecutwfc,
        ecutrho=ecutrho,
        nspin=nspin,                  # type: ignore[arg-type]
        noncolin=noncolin,
        lspinorb=lspinorb,
        starting_magnetization=spin_decision.initial_magnetization,
        angle1=spin_decision.angle1,
        angle2=spin_decision.angle2,
        vdw_corr=vdw_corr,
        smearing_decision=smearing_decision,
        kpoints_decision=kpoints_decision,
        cutoff_decision=cutoff_decision,
        spin_decision=spin_decision,
        vdw_decision=vdw_decision,
    )
