"""Translate a code-agnostic AdviceBundle into QE-specific QEParameterSet."""

from __future__ import annotations

from goldilocks_core.advise.types import AdviceBundle, QEParameterSet

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


def build_qe_parameter_set(bundle: AdviceBundle) -> QEParameterSet:
    """Translate a code-agnostic AdviceBundle into a QEParameterSet.

    All physics decisions are taken from *bundle*.  This function only
    performs unit conversion (eV → Ry) and maps physics-level names to
    QE-specific strings.
    """
    smearing = bundle.smearing
    kpoints  = bundle.kpoints
    spin     = bundle.spin
    cutoff   = bundle.cutoff
    vdw      = bundle.vdw

    # Smearing → QE SYSTEM card
    if smearing.use_smearing:
        occupations: str = "smearing"
        smearing_qe: str | None = _SMEARING_TO_QE[smearing.method]  # type: ignore[index]
        degauss = (smearing.width_ev or 0.0) * _EV_TO_RY
    else:
        occupations = "fixed"
        smearing_qe = None
        degauss = 0.0

    # Spin → QE SYSTEM card
    nspin, noncolin, lspinorb = _SPIN_TO_QE[spin.treatment]

    ecutwfc = cutoff.wavefunction_cutoff_ev * _EV_TO_RY
    ecutrho = cutoff.density_cutoff_ev * _EV_TO_RY

    vdw_corr = _VDW_TO_QE[vdw.method] if vdw.use_vdw and vdw.method else None

    return QEParameterSet(
        occupations=occupations,      # type: ignore[arg-type]
        smearing=smearing_qe,
        degauss=degauss,
        kpoints_grid=kpoints.grid,
        kpoints_shift=kpoints.shift,
        pseudos=bundle.pseudos,
        ecutwfc=ecutwfc,
        ecutrho=ecutrho,
        nspin=nspin,                  # type: ignore[arg-type]
        noncolin=noncolin,
        lspinorb=lspinorb,
        starting_magnetization=spin.initial_magnetization,
        angle1=spin.angle1,
        angle2=spin.angle2,
        vdw_corr=vdw_corr,
        smearing_decision=smearing,
        kpoints_decision=kpoints,
        cutoff_decision=cutoff,
        spin_decision=spin,
        vdw_decision=vdw,
    )
